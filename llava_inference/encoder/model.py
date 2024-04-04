import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder, ViTransformerWrapper

class vision_transformer(nn.Module):

    @beartype
    def __init__(
        self,
        image_size: int = 336,
        patch_size: int = 14,
        encoder_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_heads: int = 16,
        ):

        super().__init__()
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
            ),
        )

    @beartype
    def forward(self, x):
        x = self.encoder(x, return_embeddings=True)
        return x
