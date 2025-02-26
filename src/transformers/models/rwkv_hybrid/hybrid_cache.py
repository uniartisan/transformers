import torch
from typing import Any, Dict, Optional, Union
from transformers.cache_utils import DynamicCache


class AttnState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class FfnState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    def __init__(
        self, 
        attn_state: AttnState,
        ffn_state: FfnState
    ):
        self.attn_state = attn_state
        self.ffn_state = ffn_state

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
