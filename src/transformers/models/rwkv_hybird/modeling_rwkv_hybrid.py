from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2ForCausalLM,
        Qwen2ForQuestionAnswering,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
        Qwen2Model,
        Qwen2PreTrainedModel,
    )


import os
import threading
import gc
from .wkv import RWKV_Tmix_x070_Wrapper

import math
import torch

import torch.nn as nn
from torch.nn import functional as F
import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


## todo: from Qwen2ForCausalLM
class HybridModel(nn.Module):
    def __init__(self, rwkv_args, transformer_config):
        super(HybridModel, self).__init__()
        self.args = rwkv_args
        print(f"rwkv_args: {rwkv_args}")
        print(f"transformer_config: {transformer_config}")
        if transformer_config.tie_word_embeddings:
            transformer_config.tie_word_embeddings = False
        with no_init_weights():
            self.model = AutoModelForCausalLM.from_config(transformer_config)
        print(f"init transformer model: {self.model}")

        # Register v_first as a buffer
        self.thread_local = threading.local()
        self.thread_local.v_first = None

        # Replace the self attention to TimeMixer
        for layer_idx in range(transformer_config.num_hidden_layers):
            llama_layer = self.model.model.layers[layer_idx]
            if layer_idx in rwkv_args.layers:
                att = RWKV_Tmix_x070_Wrapper(
                    rwkv_args,
                    layer_idx,
                    self.update_v_first,  # Pass the callback function
                    self.get_v_first,
                )
                old_attn = llama_layer.self_attn
                llama_layer.self_attn = att
                del old_attn
                print(f"layer {layer_idx} is replaced by RWKV TimeMixer_x070")

        import gc

        gc.collect()

    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        # Initialize v_first as None for the first layer
        kwargs["v_first"] = None
        return self.model(input_ids, **kwargs)

    def update_v_first(self, new_v_first):
        """Callback function to update v_first in HybridModel."""
        self.thread_local.v_first = new_v_first

    def get_v_first(self):
        return self.thread_local.v_first

    def load_checkpoint(self, path):
        # FIXME！ this is a adaptor to laod channel mixing, however I think this will not be kept in the final version.
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
        else:
            files = [path]
        for file in files:
            checkpoint = torch.load(file, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'load model from {file}')
            ckpt_keys = checkpoint.keys()
            #subtract the keys in the checkpoint from the all_keys
            #if the ckpt_key exists in the all_keys, remove it
            for ckpt_key in ckpt_keys:
                if ckpt_key in all_keys:
                    all_keys.remove(ckpt_key)
                else:
                    incompatible_keys.add(ckpt_key)
            del checkpoint
            gc.collect()
        print(f'Finish loading model from {path}')
        print(f'Incompatible keys: {incompatible_keys} missing keys: {all_keys}')
        


def create_rwkv_args(transformer_config, config):
    from argparse import Namespace

    args = Namespace()
    args.layers = config["RWKV"]["layers"]
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_config.num_hidden_layers
    args.n_embd = transformer_config.hidden_size
    args.dim_att = transformer_config.hidden_size
    args.dim_ffn = transformer_config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_config.vocab_size
    args.pad_id = transformer_config.pad_token_id
    args.is_llama_ffn = config.get("is_llama_ffn", False)
    args.is_rwkv_att_only = config.get("is_rwkv_att_only", False)
    return args


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_file = "configs/qwen_7b.yaml"
    import yaml

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    from transformers import AutoConfig

    model_id = config["Llama"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_config = AutoConfig.from_pretrained(model_id)
    print(transformer_config)
    args = create_rwkv_args(transformer_config, config)
    model = HybridModel(args, transformer_config)
    print(model)
    ckpt_file = "/home/yueyulin/model/qwen_7b_distill/7b_stage2_model_converted.bin"
    model.load_checkpoint(ckpt_file)