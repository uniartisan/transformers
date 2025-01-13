import torch
from einops import rearrange

from .utilities import TimeMixState, BlockState
import math
import torch.nn as nn
from torch.nn import functional as F
from .configuration_rwkv_hybrid import RwkvHybridConfig

try:
    import triton
    from rwkvfla.ops.rwkv7 import fused_recurrent_rwkv7, native_recurrent_rwkv7  # pylint: disable=C0411
except ImportError:
    from rwkvfla.ops.rwkv7 import native_recurrent_rwkv7  # pylint: disable=C0411
    fused_recurrent_rwkv7 = native_recurrent_rwkv7

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id, update_v_first, get_v_first):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.hidden_size

        self.update_v_first = update_v_first
        self.get_v_first = get_v_first

        self.head_size = args.head_size
        self.n_head = args.num_wkv_heads
        assert args.hidden_size % self.n_head == 0
        H = self.n_head
        N = self.head_size
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - \
                (layer_id / args.num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, args.hidden_size)
            for i in range(args.hidden_size):
                ddd[0, 0, i] = i / args.hidden_size

            self.x_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) +
                       0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) +
                       0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = (
                        math.sqrt(shape[0] / shape[1]
                                  ) if shape[0] > shape[1] else 1
                    )
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = (
                        math.sqrt(shape[1] / shape[2]
                                  ) if shape[1] > shape[2] else 1
                    )
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                else:
                    assert False
                return x

            D_DECAY_LORA = 64
            self.w1 = nn.Parameter(torch.zeros(args.hidden_size, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(
                torch.zeros(D_DECAY_LORA, args.hidden_size), 0.1))
            decay_speed = torch.ones(args.hidden_size)
            for n in range(args.hidden_size):
                decay_speed[n] = -7 + 5 * (n / (args.hidden_size - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.w0 = nn.Parameter(
                decay_speed.reshape(1, 1, args.hidden_size) + 0.5
            )

            D_AAA_LORA = 64
            self.a1 = nn.Parameter(torch.zeros(args.hidden_size, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(
                torch.zeros(D_AAA_LORA, args.hidden_size), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, args.hidden_size))

            D_MV_LORA = 32
            self.v1 = nn.Parameter(torch.zeros(args.hidden_size, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(
                torch.zeros(D_MV_LORA, args.hidden_size), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, args.hidden_size) + 1.0)

            D_GATE_LORA = 128
            if self.args.wkv_has_gate:
                self.g1 = nn.Parameter(torch.zeros(
                    args.hidden_size, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(
                    torch.zeros(D_GATE_LORA, args.hidden_size), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, args.hidden_size) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, args.hidden_size))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(
                args.hidden_size, args.hidden_size, bias=False)
            self.key = nn.Linear(
                args.hidden_size, args.hidden_size, bias=False)
            self.value = nn.Linear(
                args.hidden_size, args.hidden_size, bias=False)
            self.output = nn.Linear(
                args.hidden_size, args.hidden_size, bias=False)
            if self.args.wkv_has_group_norm:
                self.ln_x = nn.GroupNorm(
                    H, args.hidden_size, eps=(
                        1e-5) * (args.head_size_divisor**2)
                )

    def apply_wkv7_state(self, r, k, v, w, a, b, s):
        r = rearrange(r, "b l (h d) -> b h l d", h=self.n_head)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_head)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_head)
        w = rearrange(w, "b l (h d) -> b h l d", h=self.n_head)
        a = rearrange(a, "b l (h d) -> b h l d", h=self.n_head)
        b = rearrange(b, "b l (h d) -> b h l d", h=self.n_head)
        wkv7_func = native_recurrent_rwkv7 if r.device == "cpu" else fused_recurrent_rwkv7
        o, state = wkv7_func(
            r,
            k,
            v,
            w,
            a,
            b,
            scale=1.0,
            initial_state=s,
            output_final_state=True,
            training=False,
        )
        x = rearrange(o, "b h l d -> b l (h d)")
        return x, state

    def forward(self, x, last_state: TimeMixState):
        shift_state = last_state.shift_state
        B, T, C = x.size()
        H = self.n_head
        if shift_state is not None:
            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
        lx = x[:, -1]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            self.update_v_first(v)
        else:
            # Original implementation
            v = v + (self.get_v_first().to(v.device) - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual

        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        if self.args.wkv_has_gate:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        wkv_state = last_state.wkv_state
        x, wkv_state = self.apply_wkv7_state(
            r, k, v, w, -kk, (kk * a), s=wkv_state,
        )
        if self.args.wkv_has_group_norm:
            x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g) if self.args.wkv_has_gate else self.output(x)
        return x, TimeMixState(lx, wkv_state)


class Rwkv7Attention(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id, update_v_first, get_v_first):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = RWKV_Tmix_x070(
            args, layer_id, update_v_first, get_v_first)
        self.HeadNums = args.hidden_size // args.head_size
        self.HeadDims = -1

    def forward(self, hidden_states, past_key_value, **kwargs):
        attn_output = hidden_states
        B, T, C = attn_output.size()
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        if last_state is None:
            wkv_states = torch.zeros(
                (B, self.HeadNums, C // self.HeadNums, C // self.HeadNums), device=attn_output.device, dtype=attn_output.dtype)
            token_shift = torch.zeros(
                (B, C), device=attn_output.device, dtype=attn_output.dtype)
            time_state = TimeMixState(token_shift, wkv_states)
            channel_state = None
            last_state = BlockState(time_state, channel_state)
        attn_output, states = self.time_mixer(
            attn_output, last_state.time_mix_state)
        last_state.time_mix_state = states

        if past_key_value is not None:
            past_key_value.update(T, last_state, self.layer_idx)
        return attn_output, None
