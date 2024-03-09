import torch
from torch import nn
from components.detr_vae import ACT_model

class act_decoder(nn.Module):

    def __init__(
        self, args
    ):
        super().__init__()
        self.decoder = ACT_model(args)

    def forward(self, src, pos, latent_input, proprio_input):
        out = self.decoder(src, pos, latent_input, proprio_input)
        return out
