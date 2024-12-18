import torch
from torch import nn
from beartype import beartype
from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper
# from flashinfer_transformer import flashinfer_decoder


class flash_palme(nn.Module):
    pass

class palme(nn.Module):

    @beartype
    def __init__(
        self,
        dim=512,
        num_text_tokens=20000,
        text_max_seq_len=2048,
        decoder_depth=6,
        attn_dim_head=64,
        attn_heads=8,
        kv_heads=2,
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
            num_tokens = num_text_tokens,
            max_seq_len = text_max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = decoder_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                kv_heads = kv_heads,
                num_mem_kv = 1,
                cross_attend = True,
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
    def forward(self, x, seq_len, context):
        x = self.wrapped_decoder.generate(x, seq_len=seq_len, context=context)
        return x