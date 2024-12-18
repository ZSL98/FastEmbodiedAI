import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder, ViTransformerWrapper

class DINOv2(nn.Module):
    # config identical to ViT-large
    @beartype
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_heads: int = 16,
        scaling: float = 1.0
        ):

        super().__init__()
        # assert encoder_depth*scaling%1==0, "vit layer num should be integer"
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=int(encoder_depth*scaling),
                heads=encoder_heads,
            ).to(torch.bfloat16),
        )

    @beartype
    def forward(self, x, slice_num=1, slice_id=0):
        x = self.encoder(x, return_embeddings=True, slice_num=slice_num, slice_id=slice_id)
        return x

