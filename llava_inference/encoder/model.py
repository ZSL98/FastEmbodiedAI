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
    def forward(self, x, slice_num=1, slice_id=0):
        x = self.encoder(x, return_embeddings=True, slice_num=slice_num, slice_id=slice_id)
        return x


class vit_sliced(nn.Module):

    @beartype
    def __init__(
        self,
        image_size: int = 336,
        patch_size: int = 14,
        encoder_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_heads: int = 16,
        sliced_id: int = 0,
        sliced_num: int = 1,
        ):

        super().__init__()
        assert encoder_depth % sliced_num == 0, f"The encoder is not divisible."

        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=int(encoder_depth/sliced_num),
                heads=encoder_heads,
            ),
        )

    @beartype
    def forward(self, x):
        x = self.encoder(x, return_embeddings=True, sliced_id=sliced_id)
        return x