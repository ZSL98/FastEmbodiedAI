import sys
sys.path.append("../../../repos/x-transformers")

import time
import torch
from torch import nn
import argparse
import itertools
from x_transformers import Encoder, ViTransformerWrapper

parser = argparse.ArgumentParser()

## profile arguments
parser.add_argument('--warmup_num', default=100, help='warmup_num', type=int)
parser.add_argument('--trail_num', default=200, help='profile trail_num', type=int)
parser.add_argument('--cuda_graph', action='store_true', help='cuda_graph')
args = parser.parse_args()

class encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTransformerWrapper(
            image_size = 256,
            patch_size = 16,
            attn_layers = Encoder(
                dim = 512,
                depth = 8,
                heads = 16
            ).to(torch.bfloat16)
        )
        self.inp = torch.randn(1, 3, 256, 256).to(torch.bfloat16).cuda()
    
    def forward(self):
        x = self.encoder(self.inp)
        return x

class encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTransformerWrapper(
            image_size = 256,
            patch_size = 16,
            attn_layers = Encoder(
                dim = 512,
                depth = 6,
                heads = 16
            ).to(torch.bfloat16)
        )
        self.inp = torch.randn(1, 3, 256, 256).to(torch.bfloat16).cuda()
    
    def forward(self):
        x = self.encoder(self.inp)

class encoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTransformerWrapper(
            image_size = 256,
            patch_size = 16,
            attn_layers = Encoder(
                dim = 512,
                depth = 24,
                heads = 16
            ).to(torch.bfloat16)
        )
        self.inp = torch.randn(1, 3, 256, 256).to(torch.bfloat16).cuda()
    
    def forward(self):
        x = self.encoder(self.inp)



class ExecOptimizer:

    def __init__(self):
        e1 = encoder1().cuda()
        e2 = encoder2().cuda()
        e3 = encoder3().cuda()
        self.modules = [e1, e2, e3]
        self.num_module = len(self.modules)
        self.graphs = [torch.cuda.CUDAGraph() for i in range(self.num_module)]

        self.original_permutation_list = [0, 1, 2]
        self.graph_stream_map = [self.modules[i] for i in self.original_permutation_list]
        self.permuted_graphs = [self.graphs[i] for i in self.original_permutation_list]

        self.streams = [torch.cuda.Stream(priority=0) for _ in range(3)]
        self.streams_high = [torch.cuda.Stream(priority=-1) for _ in range(3)]
        self.streams_higher = [torch.cuda.Stream(priority=-2) for _ in range(3)]
        self.all_streams = self.streams + self.streams_high + self.streams_higher

    def make_graph(self):
        for i in range(self.num_module):
            self.modules[i]()
            with torch.cuda.graph(self.graphs[i]):
                self.modules[i]()
            self.graphs[i].replay()
        
        torch.cuda.synchronize()

    def tuning(self):
        permutations_list = list(itertools.permutations(self.original_permutation_list))
        min_time = {}
        for perm in permutations_list:
            self.graph_stream_map = [self.modules[i] for i in perm]
            self.permuted_graphs = [self.graphs[i] for i in perm]
            assignments = itertools.product([0, 1, 2], repeat=self.num_module)
            min_time[perm] = float('inf')
            for assignment in assignments:
                self.assigned_streams = [self.streams[i] for i in assignment]
                time_res = self.profiling()
                if time_res < min_time[perm]:
                    # print("new config: {:.6f} s".format(time_res))
                    min_time[perm] = time_res
                    opt_assignment = assignment
            print("Launch jobs with sequence {} onto stream {} lasts: {:.6f} s".format(perm, opt_assignment, min_time[perm]))


    def profiling(self):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_module)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_module)]
        all_durations = [0 for i in range(self.num_module)]
        for i in range(args.trail_num + args.warmup_num):
            if i == args.warmup_num:
                start_time = time.time()

            if args.cuda_graph:
                for j, graph in enumerate(self.permuted_graphs):
                    with torch.cuda.stream(self.assigned_streams[j]):
                        start_events[j].record()
                        graph.replay()
                        end_events[j].record()
                torch.cuda.synchronize()
            else:
                for j, graph in enumerate(self.graph_stream_map):
                    with torch.cuda.stream(self.assigned_streams[j]):
                        start_events[j].record()
                        graph()
                        end_events[j].record()

            if i >= args.warmup_num:
                durations = [s.elapsed_time(e) for s, e in zip(start_events[:j+1], end_events[:j+1])]
                all_durations = [durations[i] + all_durations[i] for i in range(self.num_module)]
                # print(durations)
        
        all_durations = ["{:.4f}".format(dur/args.trail_num) for dur in all_durations]
        # print("Duration of graphs: ", all_durations)

        frame_interval = (time.time() - start_time) / args.trail_num
        # print("Frame interval: {:.8f} s".format(frame_interval))

        return frame_interval

opt = ExecOptimizer()
opt.make_graph()
opt.tuning()
opt.profiling()