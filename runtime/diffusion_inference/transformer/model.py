import torch
from torch import nn
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any

from x_transformers import Encoder, Decoder, TransformerWrapper, AutoregressiveWrapper
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


class DiffusionTransformer(nn.Module):

    # This is the configuration for the pusht task for transformer-based
    @beartype
    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        horizon=10,
        n_obs_steps=2,
        cond_dim=66,
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0
        ):

        super().__init__()
        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

    @beartype
    def forward(self, sample, timestep, cond):
        x = self.model(sample, timestep, cond)
        return x

