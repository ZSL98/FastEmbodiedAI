import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder, ViTransformerWrapper

class SigLIP(nn.Module):
    # config refs to https://huggingface.co/google/siglip-so400m-patch14-384/blob/main/config.json
    # also https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384/blob/main/open_clip_config.json
    @beartype
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_dim: int = 1152,
        encoder_depth: int = 27,
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
            ).to(torch.bfloat16),
        )

    @beartype
    def forward(self, x, slice_num=1, slice_id=0):
        x = self.encoder(x, return_embeddings=True, slice_num=slice_num, slice_id=slice_id)
        return x

