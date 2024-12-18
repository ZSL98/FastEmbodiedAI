import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder

class vision_encoder(nn.Module):

    @beartype
    def __init__(
        self,
        dim=512,
        flash_attn=True
        ):

        super().__init__()
        default_vit_kwargs = dict(
            dim = dim,
            flash_attn = flash_attn
        )
        video_encoder_kwargs = dict(
            dim = dim,
            depth = 2
        )
        self.video_encoder = Encoder(**{
            **default_vit_kwargs,
            **video_encoder_kwargs
        })

    @beartype
    def forward(self, x):
        x = self.video_encoder(x)
        return x
