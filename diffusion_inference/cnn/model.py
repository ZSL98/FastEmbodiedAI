import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

class DiffusionCNN(nn.Module):

    # This is the configuration for the pusht task for CNN-based model
    @beartype
    def __init__(
        self,
        input_dim=2,
        local_cond_dim=None,
        global_cond_dim=132,
        diffusion_step_embed_dim=128,
        down_dims=[512, 1024, 2048],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        data_type=torch.bfloat16,
        scaling=1
        ):

        super().__init__()
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            data_type=data_type
        )
    
    @beartype
    def forward(self, sample, timestep, cond):
        x = self.model(sample, timestep, None, cond)
        return x