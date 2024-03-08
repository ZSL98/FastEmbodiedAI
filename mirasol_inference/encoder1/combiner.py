import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder

class combiner(nn.Module):

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

        combiner_depth = 2
        super().__init__()
        default_combiner_kwargs = dict(
            dim = dim,
            depth = combiner_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            flash_attn = flash_attn
        )

        self.combiner = Encoder(
            **{
                **default_combiner_kwargs,
                **attn_layers_kwargs
            }
        )

    @beartype
    def forward(self, x):
        x = self.combiner(x)
        return x