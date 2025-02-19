import torch
from typing import Any, Dict, Optional, Union
from transformers.cache_utils import DynamicCache


class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class VfirstCache:
    def __init__(self, v_first: torch.Tensor):
        self.v_first = v_first


class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.zeros((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.zeros((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state


class HybridCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.rwkv_layers = set()
        self.key_cache_nums = 0
        self.v_first_cache = None

    def update(
        self,
        key_states: Union[int, torch.Tensor],
        value_states: Union[torch.Tensor, BlockState],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None
    ):
        if isinstance(key_states, int) and isinstance(value_states, BlockState):
            self.rwkv_layers.add(layer_idx)

            if layer_idx >= self.key_cache_nums:
                self.key_cache.append([])
                self.value_cache.append([])
                self.key_cache[layer_idx].append(key_states)
                self.value_cache[layer_idx].append(value_states)
                self.key_cache_nums += 1

            else:
                self.key_cache[layer_idx][0] += key_states
                self.value_cache[layer_idx][0] = value_states

            return key_states, value_states

        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def update_v_first(self, v_first: torch.Tensor):
        self.v_first_cache = v_first

    def get_v_first(self):
        return self.v_first_cache

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        if layer_idx in self.rwkv_layers:
            return self.key_cache[layer_idx][0]
        return super().get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx):
        return super().reorder_cache(beam_idx)

    def __getitem__(self, item):
        if item in self.rwkv_layers:
            return self.value_cache[item]
        return super().__getitem__(item)
