import torch
import torch.nn as nn
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("facebook/opt-13b")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# for name, param in model.named_parameters():
#     torch.nn.init.normal_(param.data, mean=0, std=0.02)  # Example initialization

# Step 2: Initialize weights manually
for name, param in model.named_parameters():
    if param.device == torch.device("meta"):  # Only initialize meta parameters
        # Get the parent module
        module_name, param_name = name.rsplit(".", 1)
        parent_module = model.get_submodule(module_name)

        # Replace the parameter with a real initialized tensor
        new_param = torch.nn.Parameter(torch.empty_like(param, device="cpu"))
        torch.nn.init.normal_(new_param.data, mean=0, std=0.02)

        # Set the initialized parameter back to the module
        setattr(parent_module, param_name, new_param)

device_map = infer_auto_device_map(model)
print(device_map)
model = dispatch_model(model, device_map)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
inputs = tokenizer("Hello, my name is", return_tensors="pt")
inputs = inputs.to(0)
output = model(inputs["input_ids"])
tokenizer.decode(output[0].tolist())