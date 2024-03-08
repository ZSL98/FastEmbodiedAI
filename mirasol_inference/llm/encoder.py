import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder

class mirasol_encoder(nn.Module):

    @beartype
    def __init__(
        self,
        dim=512,
        encoder_depth=6,
        attn_dim_head=64,
        attn_heads=8,
        attn_layers_kwargs: dict = dict(),
        flash_attn=True
        ):

        super().__init__()
        self.encoder = Encoder(
            dim = dim,
            depth = encoder_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            num_mem_kv = 1,
            flash_attn = flash_attn,
            **attn_layers_kwargs
        )

    @beartype
    def forward(self, x):
        x = self.encoder(x)
        return x