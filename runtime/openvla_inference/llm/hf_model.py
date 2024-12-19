import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

import sys
sys.path.append("./")
sys.path.append("../../third_party/x-transformers")
from model import llama

# Function to print memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
    else:
        print("CUDA is not available.")

class CustomConfig(PretrainedConfig):
    model_type = "llama"
    def __init__(
        self,
        dim=4096,
        num_text_tokens=1024,
        text_max_seq_len=4096,
        decoder_depth=32,
        attn_dim_head=128,  # 128 = dim/attn_heads = 4096/32
        attn_heads=32,
        kv_heads=8,
        attn_layers_kwargs: dict = dict(),
        flash_attn=True,
        text_forgetful_causal_mask_prob=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_text_tokens = num_text_tokens
        self.text_max_seq_len = text_max_seq_len
        self.decoder_depth = decoder_depth
        self.attn_dim_head = attn_dim_head
        self.attn_heads = attn_heads
        self.kv_heads = kv_heads
        self.attn_layers_kwargs = attn_layers_kwargs
        self.flash_attn = flash_attn
        self.text_forgetful_causal_mask_prob = text_forgetful_causal_mask_prob


class CustomModel(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config)
        self.model = llama(
            dim=config.dim,
            num_text_tokens=config.num_text_tokens,
            text_max_seq_len=config.text_max_seq_len,
            decoder_depth=config.decoder_depth,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            kv_heads=config.kv_heads,
            attn_layers_kwargs=config.attn_layers_kwargs,
            flash_attn=config.flash_attn,
            text_forgetful_causal_mask_prob=config.text_forgetful_causal_mask_prob
        )
    def forward(self, *inputs, **kwargs):
        return self.model.decoder(*inputs, **kwargs)
    
class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(
        self,
        dim=4096,
        num_text_tokens=1024,
        text_max_seq_len=4096,
        decoder_depth=32,
        attn_dim_head=128,
        attn_heads=32,
        kv_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_text_tokens = num_text_tokens
        self.text_max_seq_len = text_max_seq_len
        self.decoder_depth = decoder_depth
        self.attn_dim_head = attn_dim_head
        self.attn_heads = attn_heads
        self.kv_heads = kv_heads


class CustomTransformerModel(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config):
        super().__init__(config)

        # Token Embeddings
        self.token_embedding = nn.Embedding(config.num_text_tokens, config.dim)

        # Positional Embeddings
        self.pos_embedding = nn.Embedding(config.text_max_seq_len, config.dim)

        # Decoder Layers (Self-Attention Only)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.attn_heads,
                dim_feedforward=config.dim * 4,
                batch_first=True
            )
            for _ in range(config.decoder_depth)
        ])

        # Final Layer Norm
        self.layer_norm = nn.LayerNorm(config.dim)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Create token embeddings
        token_embeddings = self.token_embedding(input_ids)

        # Create positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embedding(position_ids)

        # Combine token and positional embeddings
        x = token_embeddings + position_embeddings

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x)  # Self-attention only (no memory input)

        # Final normalization
        x = self.layer_norm(x)
        return x



# config = CustomConfig()
# model = CustomModel(config)

# inputs = torch.randint(0, 1024, (1, 4096))
# # seq_len = torch.tensor([4096])
# outputs = model(inputs)
# print(outputs)

# Create configuration
print_gpu_memory()
config = CustomTransformerConfig(
    dim=4096,
    num_text_tokens=1024,
    text_max_seq_len=4096,
    decoder_depth=4,
    attn_dim_head=128,
    attn_heads=32,
    kv_heads=8
)

# Instantiate the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CustomTransformerModel(config).to(device)
print_gpu_memory()
# Example input: Batch of token IDs (batch_size=2, seq_len=4096)
model = model.to(dtype=torch.float16)
print_gpu_memory()

input_ids = torch.randint(0, config.num_text_tokens, (2, 4096)).to(device)
print_gpu_memory()
# Forward pass


output = model(input_ids)
print("Output shape:", output.shape)

from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
# model2 = CustomTransformerModel(config)
# # model2 = model2.to(dtype=torch.float16)
# device_map = infer_auto_device_map(model2, max_memory={0: "1GiB", "cpu": "4GiB"})
# model2 = dispatch_model(model2, device_map=device_map)
# input_device = next(model2.parameters()).device
# input_ids2 = torch.randint(0, config.num_text_tokens, (2, 4096)).to(input_device)
# output = model2(input_ids2)

with init_empty_weights():
    model2 = CustomTransformerModel(config)

# Step 2: Initialize weights manually
for name, param in model2.named_parameters():
    if param.device == torch.device("meta"):  # Only initialize meta parameters
        # Get the parent module
        module_name, param_name = name.rsplit(".", 1)
        parent_module = model2.get_submodule(module_name)

        # Replace the parameter with a real initialized tensor
        new_param = torch.nn.Parameter(torch.empty_like(param, device="cpu"))
        torch.nn.init.normal_(new_param.data, mean=0, std=0.02)

        # Set the initialized parameter back to the module
        setattr(parent_module, param_name, new_param)

device_map = infer_auto_device_map(model2)
print(f"Device map: {device_map}")

model2 = dispatch_model(model2, device_map=device_map)
input_device = next(model2.parameters()).device
input_ids2 = torch.randint(0, config.num_text_tokens, (2, 4096)).to(input_device)

output = model2(input_ids2)

print_gpu_memory()
print("Output shape:", output.shape)