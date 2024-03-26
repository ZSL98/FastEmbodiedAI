import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import flashinfer

from einops import rearrange
from sche_plan import args
import time

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

class LayerNorm(nn.Module):
    def __init__(self, dim):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


class FIAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        value_dim_head = None,

    ):
        super().__init__()

        dim_kv = default(dim_context, dim)
        self.heads = heads

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        # kv_heads = 1 if one_kv_head else kv_heads
        # assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, k_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, v_dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        cache = None
    ):
        q = self.to_q(x)

        # self-attention
        if exists(cache) and context is None:
            k_append = self.to_k(x)
            v_append = self.to_v(x)
            k_append = rearrange(k_append, 'b (h w) -> b 1 h w', h = 8)  # num_kv_heads = 8
            v_append = rearrange(v_append, 'b (h w) -> b 1 h w', h = 8)  # num_kv_heads = 8
            k = torch.cat((cache[0], k_append), dim = -3)
            v = torch.cat((cache[1], v_append), dim = -3)
        # cross-attention
        elif exists(context):
            k = context[0]
            v = context[1]

        # print("Q shape: ", q.shape)
        q = rearrange(q, 'b (h w) -> b h w', h = 8)  # num_qo_heads = 8
        # k = rearrange(k, 'b c (h w) -> b c h w', h = 8)  # num_kv_heads = 8
        # v = rearrange(v, 'b c (h w) -> b c h w', h = 8)  # num_kv_heads = 8

        o = flashinfer.batch_decode_with_padded_kv_cache(q, k, v, "NHD")

        return o

class FIAttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        cross,
        heads = 8,
    ):
        super().__init__()
        if cross:
            default_block = ('a', 'c', 'f')
        else:
            default_block = ('a', 'f')

        layer_types = default_block * depth
        self.layers = nn.ModuleList([])
        self.layer_types = layer_types
        self.layer_dropouts = cast_tuple(0, len(layer_types))
        for ind, layer_type in enumerate(self.layer_types):
            if layer_type == 'a':
                layer = FIAttention(dim, heads = heads)
            elif layer_type == 'c':
                layer = FIAttention(dim, heads = heads)
            elif layer_type == 'f':
                layer = FeedForward(dim)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual = Residual(dim)
            norm = LayerNorm(dim)
            self.layers.append(nn.ModuleList([
                norm,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        context,
        cache
    ):
        layer_variables = (
            self.layer_types,
            self.layers,
            self.layer_dropouts
        )

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            if layer_type == 'a':
                out = block(x, cache = cache)
            elif layer_type == 'c':
                out = block(x, context = context)
            elif layer_type == 'f':
                out = block(x)

        return out


def build_ts(task_plan, ts_decode_num, ts_prefill_num):
    cross = True

    padded_kv_len = args.input_seq_len * ts_decode_num
    decode_model = FIAttentionLayers(
        dim=args.dim,
        depth=6,
        cross=cross,
    ).half().to("cuda:0")

    return decode_model

def build_graph(ts_model, ts_decode_num, ts_prefill_num):

    padded_kv_len = args.input_seq_len
    num_qo_heads = 8
    num_kv_heads = 8
    batch_size = ts_decode_num
    head_dim = 64

    context_kv_len = 18

    # q = torch.randn(batch_size, num_qo_heads, head_dim).to("cuda:0")
    input_q = torch.randn(batch_size, args.dim).half().to("cuda:0")
    k_cache = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    v_cache = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")

    k_context = torch.randn(batch_size, context_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    v_context = torch.randn(batch_size, context_kv_len, num_kv_heads, head_dim).half().to("cuda:0")

    out = ts_model(input_q, context=[k_context, v_context], cache=[k_cache, v_cache])
    ts_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ts_graph):
        out = ts_model(input_q, context=[k_context, v_context], cache=[k_cache, v_cache])

    ts_graph.replay()
    return ts_graph


def profile_graph(ts_graph):
    for i in range(args.warmup_num):
        ts_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(args.trail_num):
        ts_graph.replay()
    torch.cuda.synchronize()
    duration = (time.time() - start) * 1000 / args.trail_num
    print("Avg graph duration: {:.2f} ms".format(duration))

def flashinfer_decode(sche_plan):
    # for task_plan in sche_plan:
    task_plan = None

    ts_decode_num = 5 # This is the batch size
    ts_prefill_num = 1
    ts_model = build_ts(task_plan, ts_decode_num, ts_prefill_num)
    ts_graph = build_graph(ts_model, ts_decode_num, ts_prefill_num)

    profile_graph(ts_graph)