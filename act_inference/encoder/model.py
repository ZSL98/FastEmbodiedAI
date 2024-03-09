import torch
from torch import nn
from components.detr_vae import ACT_vision_encoder

class act_encoder(nn.Module):

    def __init__(
        self, args
    ):
        super().__init__()
        self.encoder = ACT_vision_encoder(args)

    def forward(self, qpos, camera_imgs):
        src, pos, latent_input, proprio_input = self.encoder(qpos, camera_imgs)
        return src, pos, latent_input, proprio_input
