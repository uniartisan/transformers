import torch
from einops import rearrange

from .hybrid_cache import TimeMixState, BlockState
import math
import torch.nn as nn
from torch.nn import functional as F
from .configuration_rwkv_hybrid import RwkvHybridConfig
from typing import Optional
from .hybrid_cache import HybridCache

try:
    import triton  # pylint: disable=F401
    from rwkvfla.ops.rwkv7 import (
        fused_recurrent_rwkv7,
        chunk_rwkv7,
        native_recurrent_rwkv7,
        fused_addcmul_rwkv7,
    )  # pylint: disable=C0411
    from rwkvfla.ops.rwkv6 import (
        fused_recurrent_rwkv6,
        chunk_rwkv6,
        native_recurrent_rwkv6,
    )
except ImportError:
    from rwkvfla.ops.rwkv7 import native_recurrent_rwkv7  # pylint: disable=C0411
    from rwkvfla.ops.rwkv6 import native_recurrent_rwkv6
    from rwkvfla.ops.rwkv7 import torch_addcmul_rwkv7

    fused_recurrent_rwkv7 = native_recurrent_rwkv7
    chunk_rwkv7 = native_recurrent_rwkv7
    chunk_rwkv6 = native_recurrent_rwkv6
    fused_recurrent_rwkv6 = native_recurrent_rwkv6
    fused_addcmul_rwkv7 = torch_addcmul_rwkv7

from rwkvfla.utils import check_pytorch_version

if check_pytorch_version("2.6"):
    compile_decorator = torch.compile
    torch._dynamo.config.cache_size_limit = 512
else:
    def compile_decorator(func):
        return func


class Rwkv_Tmix_x070(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id, **kwargs):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size

        self.head_size = args.head_size
        self.n_head = args.num_wkv_heads
        assert args.hidden_size % self.n_head == 0
        H = self.n_head
        N = self.head_size

        self.x_r = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.x_w = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.x_k = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.x_v = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.x_a = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        D_DECAY_LORA = 64
        D_AAA_LORA = 64
        D_MV_LORA = 32
        D_GATE_LORA = 128

        self.w1 = nn.Parameter(torch.Tensor(args.hidden_size, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.Tensor(D_DECAY_LORA, args.hidden_size))
        self.w0 = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        self.a1 = nn.Parameter(torch.Tensor(args.hidden_size, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.Tensor(D_AAA_LORA, args.hidden_size))
        self.a0 = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        self.v1 = nn.Parameter(torch.Tensor(args.hidden_size, D_MV_LORA))
        self.v2 = nn.Parameter(torch.Tensor(D_MV_LORA, args.hidden_size))
        self.v0 = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))

        if self.args.wkv_has_gate:
            self.x_g = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
            self.g1 = nn.Parameter(torch.Tensor(args.hidden_size, D_GATE_LORA))
            self.g2 = nn.Parameter(torch.Tensor(D_GATE_LORA, args.hidden_size))

        self.k_k = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.k_a = nn.Parameter(torch.Tensor(1, 1, args.hidden_size))
        self.r_k = nn.Parameter(torch.Tensor(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(
            args.hidden_size, args.hidden_size, bias=False)
        self.key = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.value = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.output = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        if self.args.wkv_has_group_norm:
            self.ln_x = nn.GroupNorm(
                H, args.hidden_size, eps=(1e-5) * (args.head_size_divisor**2)
            )

    def post_init(self):
        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / \
                (self.args.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (
                self.layer_id / self.args.num_hidden_layers
            )  # 1 to ~0

            ddd = torch.ones(1, 1, self.args.hidden_size)
            for i in range(self.args.hidden_size):
                ddd[0, 0, i] = i / self.args.hidden_size

            nn.init.constant_(
                self.x_r, 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            nn.init.constant_(
                self.x_w, 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            nn.init.constant_(
                self.x_k,
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) +
                       0.4 * ratio_0_to_1),
            )
            nn.init.constant_(
                self.x_v,
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) +
                       0.6 * ratio_0_to_1),
            )
            nn.init.constant_(
                self.x_a, 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                shape = x.shape
                original_dtype = x.dtype
                x_fp32 = x.float()
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]
                                     ) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x_fp32, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]
                                     ) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x_fp32[i], gain=gain * scale)
                else:
                    raise ValueError(
                        "ortho_init only supports 2D or 3D tensors")
                x.data.copy_(x_fp32.to(original_dtype))
                return x

            D_DECAY_LORA = 64
            nn.init.zeros_(self.w1)
            self.w2 = nn.Parameter(
                ortho_init(torch.zeros(
                    D_DECAY_LORA, self.args.hidden_size), 0.1)
            )

            decay_speed = torch.ones(self.args.hidden_size)
            for n in range(self.args.hidden_size):
                decay_speed[n] = -7 + 5 * (n / (self.args.hidden_size - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            nn.init.constant_(
                self.w0, decay_speed.reshape(1, 1, self.args.hidden_size) + 0.5
            )

            D_AAA_LORA = 64
            nn.init.zeros_(self.a1)
            self.a2 = nn.Parameter(
                ortho_init(torch.zeros(D_AAA_LORA, self.args.hidden_size), 0.1)
            )
            nn.init.zeros_(self.a0)

            D_MV_LORA = 32
            nn.init.zeros_(self.v1)
            self.v2 = nn.Parameter(
                ortho_init(torch.zeros(D_MV_LORA, self.args.hidden_size), 0.1)
            )
            nn.init.constant_(self.v0, 1.0)

            D_GATE_LORA = 128
            if self.args.wkv_has_gate:
                nn.init.zeros_(self.g1)
                self.g2 = nn.Parameter(
                    ortho_init(torch.zeros(
                        D_GATE_LORA, self.args.hidden_size), 0.1)
                )
                nn.init.constant_(
                    self.x_g, 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            nn.init.constant_(self.k_k, 0.85)
            nn.init.constant_(self.k_a, 1.0)
            nn.init.zeros_(self.r_k)

            nn.init.zeros_(self.receptance.weight)
            nn.init.zeros_(self.key.weight)
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.output.weight)

            if self.args.wkv_has_group_norm:
                nn.init.ones_(self.ln_x.weight)
                nn.init.zeros_(self.ln_x.bias)

    def apply_wkv7_state(
        self, r, k, v, w, a, b, s,
        output_final_state,
        cu_seqlens
    ):
        if r.device.type == "cpu":
            r, w, k, v, a, b = map(lambda x: rearrange(
                x, 'b l (h d) -> b h l d', h=self.n_head), (r, w, k, v, a, b))
            o, state = native_recurrent_rwkv7(
                r=r, k=k, v=v, w=w,
                a=a, b=b,
                scale=1.0,
                initial_state=s.transpose(-1, -2),
                output_final_state=True,
                head_first=True,
            )
            state = state.transpose(-1, -2)
            x = rearrange(o, "b h l d -> b l (h d)")
        else:
            r, w, k, v, a, b = map(lambda x: rearrange(
                x, 'b l (h d) -> b l h d', h=self.n_head), (r, w, k, v, a, b))
            wkv7_func = chunk_rwkv7 if r.shape[1] != 1 else fused_recurrent_rwkv7
            o, state = wkv7_func(
                r=r, k=k, v=v, w=w,
                a=a, b=b,
                scale=1.0,
                initial_state=s,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_first=False,
            )
            x = rearrange(o, "b l h d -> b l (h d)")
        return x, state

    @compile_decorator
    def forward(
            self,
            hidden_states,
            last_state: TimeMixState,
            use_cache: Optional[bool] = False,
            cu_seqlens: Optional[torch.Tensor] = None,
            v_first: Optional[torch.Tensor] = None,
            **kwargs
    ):
        shift_state = last_state.shift_state
        B, T, C = hidden_states.size()

        xx = torch.concat((shift_state.unsqueeze(
            1), hidden_states[:, :-1]), dim=1) - hidden_states

        lx = hidden_states[:, -1]

        if self.args.wkv_has_gate:
            xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
                hidden_states, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        else:
            xr, xw, xk, xv, xa, _ = fused_addcmul_rwkv7(
                hidden_states, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a)

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            ))  # add value residual

        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        if self.args.wkv_has_gate:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, self.n_head, -1),
                         p=2.0, dim=-1, eps=1e-4 if kk.dtype == torch.float16 else 1e-12).view(B, T, C)
        k = torch.lerp(k, k * a, self.k_a)

        wkv_state = last_state.wkv_state
        hidden_states, wkv_state = self.apply_wkv7_state(
            r,
            k,
            v,
            w,
            -kk,
            (kk * a),
            s=wkv_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        if self.args.wkv_has_group_norm:
            hidden_states = self.ln_x(
                hidden_states.view(B * T, C)).view(B, T, C)

        # original code:
        # weighted_sum_rk = (r.view(B, T, self.n_head, -1) * k.view(B, T, self.n_head, -1) * self.r_k).sum(
        #         dim=-1, keepdim=True
        #     )
        weighted_sum_rk = torch.einsum('btij,btij,ij->btij', r.view(B, T, self.n_head, -1),
                                       k.view(B, T, self.n_head, -1), self.r_k).sum(dim=-1, keepdim=True)
        hidden_states = hidden_states + \
            (weighted_sum_rk * v.view(B, T, self.n_head, -1)).view(B, T, C)
        hidden_states = self.output(
            hidden_states * g) if self.args.wkv_has_gate else self.output(hidden_states)
        return hidden_states, TimeMixState(lx, wkv_state), v_first


class Rwkv7Attention(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = Rwkv_Tmix_x070(args, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[HybridCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        **kwargs
    ):

        batch_size, token_length, _ = hidden_states.shape

        if use_cache and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx][0]
        else:
            last_state = self.init_state(
                batch_size, hidden_states.device, hidden_states.dtype
            )

        attn_output, states, v_first = self.time_mixer(hidden_states=hidden_states,
                                                       last_state=last_state.time_mix_state,
                                                       use_cache=use_cache,
                                                       cu_seqlens=cu_seqlens,
                                                       v_first=v_first,
                                                       **kwargs)

        if use_cache:
            last_state.time_mix_state = states
            past_key_value.update(token_length, last_state, self.layer_idx)

        return attn_output, None, v_first

    def init_state(self, batch_size, device, dtype) -> BlockState:
        wkv_states = torch.zeros(
            (
                batch_size,
                self.args.num_wkv_heads,
                self.args.head_size,
                self.args.head_size,
            ),
            device=device,
            dtype=torch.float32,
        )
        shift_states = torch.zeros(
            (batch_size, self.args.hidden_size), device=device, dtype=dtype
        )
        return BlockState(TimeMixState(shift_states, wkv_states), None)


class Rwkv_Tmix_x060(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id, **kwargs):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size

        self.head_size = args.head_size
        self.n_head = args.num_wkv_heads
        assert args.hidden_size % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.hidden_size)
            for i in range(args.hidden_size):
                ddd[0, 0, i] = i / args.hidden_size

            # fancy time_mix
            self.time_maa_x = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
            if args.hidden_size == 4096:
                D_MIX_LORA = D_MIX_LORA * 2
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(args.hidden_size, D_MIX_LORA * 5)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_LORA,
                            args.hidden_size).uniform_(-0.01, 0.01)
            )

            # fancy time_decay
            decay_speed = torch.ones(args.head_size)
            for n in range(args.head_size):
                decay_speed[n] = -6 + 5 * (n / (args.head_size - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(
                decay_speed.reshape(1, 1, args.head_size))

            D_DECAY_LORA = 64
            if args.hidden_size == 4096:
                D_DECAY_LORA = D_DECAY_LORA * 2
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(args.hidden_size, D_DECAY_LORA)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(D_DECAY_LORA, args.head_size).uniform_(-0.01, 0.01)
            )

            tmp = torch.zeros(args.head_size)
            for n in range(args.head_size):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * \
                    (1 - (n / (args.head_size - 1))) + zigzag

            self.time_faaaa = nn.Parameter(
                tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(
            args.hidden_size, args.head_size, bias=False)
        self.key = nn.Linear(args.hidden_size, args.head_size, bias=False)

        self.value = nn.Linear(args.hidden_size, args.head_size, bias=False)
        self.output = nn.Linear(args.head_size, args.hidden_size, bias=False)
        self.gate = nn.Linear(args.hidden_size, args.head_size, bias=False)

        if self.args.wkv_has_group_norm:
            self.ln_x = nn.GroupNorm(
                self.n_head, args.head_size, eps=(
                    1e-5) * (args.head_size_divisor**2)
            )

    def post_init(self):
        pass

    @compile_decorator
    def forward(
        self,
        hidden_states,
        last_state: TimeMixState,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        **kwargs
    ):
        shift_state = last_state.shift_state
        B, T, C = hidden_states.size()
        H = self.n_head

        xx = torch.concat((shift_state.unsqueeze(
            1), hidden_states[:, :-1]), dim=1) - hidden_states

        lx = hidden_states[:, -1]

        xxx = hidden_states + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B *
                                                      T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = hidden_states + xx * (self.time_maa_w + mw)
        xk = hidden_states + xx * (self.time_maa_k + mk)
        xv = hidden_states + xx * (self.time_maa_v + mv)
        xr = hidden_states + xx * (self.time_maa_r + mr)
        xg = hidden_states + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        wkv_state = last_state.wkv_state
        hidden_states, wkv_state = self.apply_wkv6_state(
            B, T, C, H, r, k, v, w, u=self.time_faaaa, s=wkv_state
        )
        if self.args.wkv_has_group_norm:
            hidden_states = self.ln_x(
                hidden_states.view(B * T, C)).view(B, T, C)
        hidden_states = self.output(hidden_states * g)
        return hidden_states, TimeMixState(lx, wkv_state)

    def apply_wkv6_state(self, B, T, C, H, r, k, v, w, u, s):
        r, w, k, v = map(lambda x: rearrange(
            x, 'b l (h d) -> b h l d', h=self.n_head), (r, w, k, v))

        if r.device.type == "cpu":
            wkv6_func = native_recurrent_rwkv6
        elif self.training:
            wkv6_func = chunk_rwkv6
        else:
            wkv6_func = fused_recurrent_rwkv6

        o, state = wkv6_func(
            r,
            k,
            v,
            -torch.exp(w),
            u=u,
            scale=1.0,
            initial_state=s,
            output_final_state=True,
        )
        x = rearrange(o, "b h l d -> b l (h d)")
        return x, state


class Rwkv6Attention(nn.Module):
    def __init__(self, args: RwkvHybridConfig, layer_id, **kwargs):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = Rwkv_Tmix_x060(args, layer_id, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[HybridCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        **kwargs
    ):
        attn_output = hidden_states
        B, T, C = attn_output.size()
        if use_cache:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        if last_state is None:
            wkv_states = torch.zeros(
                (B, self.args.num_wkv_heads,
                 self.args.head_size, self.args.head_size),
                device=attn_output.device,
                dtype=torch.float32,
            )
            token_shift = torch.zeros(
                (B, C), device=attn_output.device, dtype=attn_output.dtype
            )
            time_state = TimeMixState(token_shift, wkv_states)
            channel_state = None
            last_state = BlockState(time_state, channel_state)
        attn_output, states = self.time_mixer(
            attn_output, last_state.time_mix_state)
        last_state.time_mix_state = states

        if use_cache:
            past_key_value.update(T, last_state, self.layer_idx)
        return attn_output, None, None
