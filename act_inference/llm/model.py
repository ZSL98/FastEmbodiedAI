import torch
from components.detr_vae import build_ACT_model

class act_decoder(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.decoder = build_ACT_model()

    def forward(self, x):
        x = self.decoder(x)
        return x
