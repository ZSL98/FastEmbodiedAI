import torch
import torch.nn as nn
from cuda import cuda

inl = torch.rand(128, 128, device="cuda")
lin = nn.Linear(128, 128, device='cuda')

inc = torch.ones(50, 30, 10, 5, device="cuda").share_memory_()
conv = nn.Conv2d(30, 5, 3, stride=1, padding=1, device='cuda').share_memory()

def create_context(sm_count):
	affinity = cuda.CUexecAffinityParam()
	affinity.type = cuda.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT
	affinity.param.smCount.val = sm_count

	ctx = cuda.cuCtxCreate_v3([affinity], 1, 0, 0)[1]
	cuda.cuInit(0)
	
	return ctx

# Creating two more contexts
ctx1 = create_context(10)
ctx2 = create_context(40)

# Trying Fully Connected layer
cuda.cuCtxSetCurrent(0) # Sets default context
dummy = lin(inl)

cuda.cuCtxSetCurrent(ctx1)
dummy = lin(inl)

cuda.cuCtxSetCurrent(ctx2)
dummy = lin(inl)

# Trying with Convolutional layer
cuda.cuCtxSetCurrent(0) # Sets default context
dummy = conv(inc)

cuda.cuCtxSetCurrent(ctx1)
dummy = conv(inc)

cuda.cuCtxSetCurrent(ctx2)
dummy = conv(inc)