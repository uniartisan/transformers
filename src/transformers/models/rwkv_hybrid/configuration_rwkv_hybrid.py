# coding=utf-8
# Copyright 2025 RWKV team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RwkvHybrid model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging
from typing import Optional, Union, List


logger = logging.get_logger(__name__)


class RwkvHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RwkvHybridModel`]. It is used to instantiate a
    RwkvHybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    RwkvHybrid-7B-beta.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the RwkvHybrid model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RwkvHybridModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_size (`int`, *optional*, defaults to 64):
            Dimensionality of each RWKV attention head. Defines the hidden dimension size for RWKV attention mechanisms.
        head_size_divisor (`int`, *optional*, defaults to 8):
            Constraint for head_size initialization, typically set to the square root of head_size. Ensures divisibility
            between hidden_size and head_size.
        wkv_version (`int`, *optional*, defaults to 7):
            Version of RWKV attention implementation. Currently supports:
            - 6: Original implementation requiring `wkv_has_gate=True` and `wkv_use_vfirst=False`
            - 7: Improved version requiring `wkv_use_vfirst=True`
        wkv_has_gate (`bool`, *optional*, defaults to False):
            Whether to include gating mechanism in RWKV attention. Required for version 6.
        wkv_has_group_norm (`bool`, *optional*, defaults to True):
            Whether to apply group normalization in RWKV attention layers.
        wkv_use_vfirst (`bool`, *optional*, defaults to True):
            Whether to prioritize value projection in RWKV attention computation. Required for version 7.
        wkv_layers (`Union[str, List[int]]`, *optional*, defaults to None):
            Specifies which layers use RWKV attention:
            - `"full"` or `None`: All layers use RWKV
            - List of integers: Only specified layers (e.g., `[0,1,2]`) use RWKV attention

    ```python
    >>> from transformers import RwkvHybridModel, RwkvHybridConfig

    >>> # Initializing a RwkvHybrid style configuration
    >>> configuration = RwkvHybridConfig()

    >>> # Initializing a model from the RwkvHybrid-7B style configuration
    >>> model = RwkvHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwkv_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `RwkvHybrid`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_size: int = 64,
        head_size_divisor: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        wkv_version: int = 7,
        wkv_has_gate: bool = False,
        wkv_has_group_norm: bool = True,
        wkv_use_vfirst: bool = True,
        wkv_layers: Optional[Union[str, List[int]]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_wkv_heads = hidden_size // head_size
        assert hidden_size % head_size == 0, "hidden_size must be divisible by head_size"
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor
        self.wkv_version = wkv_version

        self.wkv_has_gate = wkv_has_gate
        self.wkv_has_group_norm = wkv_has_group_norm
        self.wkv_use_vfirst = wkv_use_vfirst

        if self.wkv_version == 7:
            assert self.wkv_use_vfirst, "wkv_use_vfirst must be True for wkv_version 7"
        elif self.wkv_version == 6:
            assert self.wkv_has_gate, "wkv_has_gate must be True for wkv_version 6"
            assert not self.wkv_use_vfirst, "wkv_use_vfirst must be False for wkv_version 6"
        else:
            raise NotImplementedError(f"Unsupported wkv_version: {self.wkv_version}, \
                                        wkv_version must be 6 or 7")

        if wkv_layers == "full" or wkv_layers == None:
            self.wkv_layers = list(range(num_hidden_layers))
        elif isinstance(wkv_layers, list):
            if all(isinstance(layer, int) for layer in wkv_layers):
                self.wkv_layers = wkv_layers
            else:
                raise ValueError(
                    "All elements in wkv_layers must be integers.")
        else:
            raise TypeError(
                "wkv_layers must be either 'full', None, or a list of integers.")

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
