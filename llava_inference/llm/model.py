import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder, Decoder, TransformerWrapper, AutoregressiveWrapper

class llama(nn.Module):

    @beartype
    def __init__(
        self,
        dim=4096,
        num_text_tokens=1024,
        text_max_seq_len=4096,
        decoder_depth=32,
        attn_dim_head=128,  # 128 = dim/attn_heads = 4096/32
        attn_heads=32,
        attn_layers_kwargs: dict = dict(),
        flash_attn=True,
        text_forgetful_causal_mask_prob=0.1,
        autoregressive_wrapper_kwargs: dict = dict(
            pad_value = 0,
            ignore_index = -100
        )
        ):

        super().__init__()
        self.decoder = TransformerWrapper(
            num_tokens = num_text_tokens + 1,
            max_seq_len = text_max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = decoder_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                num_mem_kv = 1,
                cross_attend = False,
                rotary_pos_emb = True,
                flash_attn = flash_attn,
                **attn_layers_kwargs
            )
        )

        self.wrapped_decoder = AutoregressiveWrapper(
            self.decoder,
            mask_prob = text_forgetful_causal_mask_prob,
            **autoregressive_wrapper_kwargs
        )

    @beartype
    def forward(self, x, seq_len):
        x = self.wrapped_decoder.generate(x, seq_len=seq_len)
        return x