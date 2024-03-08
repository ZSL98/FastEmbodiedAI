import torch
from components.detr_vae import build_ACT_encoder

class act_encoder(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = build_ACT_encoder()

    def forward(self, x):
        x = self.encoder(x)
        return x
