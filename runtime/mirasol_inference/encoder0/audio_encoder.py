import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder

class audio_encoder(nn.Module):

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
        audio_encoder_kwargs = dict(
            dim = dim,
            depth = 2
        )
        self.audio_encoder = Encoder(**{
            **default_vit_kwargs,
            **audio_encoder_kwargs
        })

    @beartype
    def forward(self, x):
        x = self.audio_encoder(x)
        return x




