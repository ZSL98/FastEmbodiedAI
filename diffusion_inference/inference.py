import torch
import torchvision.models as models
import numpy as np

import os
import sys
import time
import math
import threading
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Value, Array

import torch.cuda.profiler as profiler
# from cuda import cuda, cudart

from .transformer.model import DiffusionTransformer
from .cnn.model import DiffusionCNN

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args, pattern_analyze


class Diffusion_engine:
    # [TODO]: The projection and tokenizers are omitted in this implementation!

    # The configurations of the Diffusion Policy (Transformer-based) [2024.11.25]:

    def __init__(self, 
                 idx = 0,
                 model_type='transformer'):
        self.idx = idx

        # prepare caches for tensors
        if args.mode == 'seq':
            args.diffusion_stage_num = 1

        img = torch.randn(2, 3, 96, 96).to(torch.bfloat16).to("cuda")
        if model_type == 'transformer':
            trajectory = [torch.randn(1, 10, 2).to(torch.bfloat16).to("cuda") for i in range(args.diffusion_stage_num)]
            cond = [torch.randn(1, 2, 66).to(torch.bfloat16).to("cuda") for i in range(args.diffusion_stage_num)]
        elif model_type == 'cnn':
            trajectory = [torch.randn(1, 16, 2).to(torch.bfloat16).to("cuda") for i in range(args.diffusion_stage_num)]
            cond = [torch.randn(1, 132).to(torch.bfloat16).to("cuda") for i in range(args.diffusion_stage_num)]     
        timestep = [torch.tensor([0], dtype=torch.long, device="cuda") for i in range(args.diffusion_stage_num)]
        self.caches = { 'img': img,
                        'trajectory': trajectory,
                        'timestep': timestep,
                        'cond': cond}

        # prepare models
        resnet = models.resnet18().to(torch.bfloat16).to("cuda")
        if model_type == 'transformer':
            backbone = DiffusionTransformer().to(torch.bfloat16).to("cuda")
        elif model_type == 'cnn':
            backbone = DiffusionCNN().to(torch.bfloat16).to("cuda")

        self.models = {'resnet': resnet,
                       'backbone': backbone}

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(36)]

        # prepare cuda graphs
        self.graphs = {'resnet': torch.cuda.CUDAGraph(),
                        'backbone': [torch.cuda.CUDAGraph() for i in range(args.diffusion_stage_num)]}
        self.generate_cuda_graphs()
        self.ours_graphs = {}


    def generate_cuda_graphs(self):

        ## Make cuda graph for the vision encoder
        self.resnet_out = self.models['resnet'](self.caches['img'])
        with torch.cuda.graph(self.graphs['resnet'], stream=self.streams[0]):
            self.resnet_out = self.models['resnet'](self.caches['img'])
        torch.cuda.synchronize()
        print("====== Graph for resnet generated ======")

        self.backbone_out = self.models['backbone'](self.caches['trajectory'][0], self.caches['timestep'][0], self.caches['cond'][0])
        self.backbone_out = {}
        for stage_id in range(args.diffusion_stage_num):
            with torch.cuda.graph(self.graphs['backbone'][stage_id], stream=self.streams[stage_id+1]):
                self.backbone_out[stage_id] = self.models['backbone'](self.caches['trajectory'][stage_id], 
                                                                            self.caches['timestep'][stage_id], 
                                                                            self.caches['cond'][stage_id])
            
        torch.cuda.synchronize()
        print("====== Graph for backbone generated ======")


    def run_cuda_graphs(self, num_trails):
        for i in range(num_trails):
            self.graphs['resnet'].replay()
            for diffusion_step in range(args.diffusion_step):
                self.graphs['backbone'][0].replay()
        torch.cuda.synchronize()


    def run_V_cuda_graphs(self, num_trails=1, required_sync=True):
        for i in range(num_trails):
            self.graphs['resnet'].replay()
            if required_sync:
                torch.cuda.synchronize()


    def run_L_cuda_graphs(self, num_trails=1, diffusion_step=64, required_sync=True):
        for i in range(num_trails):
            for step in range(diffusion_step):
                self.graphs['backbone'][0].replay()
                if required_sync:
                    torch.cuda.synchronize()


    def run_VL_ms(self):
        with torch.cuda.stream(self.streams[0]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
        with torch.cuda.stream(self.streams[1]):
            self.run_L_cuda_graphs(num_trails=1, diffusion_step=args.diffusion_step//args.diffusion_stage_num, required_sync=False)



    def run_single_request(self, durations, i):
        worker_num = 2
        stream_id = i%worker_num
        req_start = time.time()

        with torch.cuda.stream(self.streams[stream_id]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
            self.run_L_cuda_graphs(num_trails=1, diffusion_step=args.diffusion_step//args.diffusion_stage_num, required_sync=False)
        
        self.streams[stream_id].synchronize()
        duration = time.time() - req_start
        print("Request duration: {:.3f} ms".format(duration*1000))
        durations.append(time.time() - req_start)


    def run_single_request_v2(self, stream, start_event, end_event, required_sync=False):
        req_start = time.time()
        # print("Launch - {}".format(self.idx))
        
        with torch.cuda.stream(stream):
            start_event.record()
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
            self.run_L_cuda_graphs(num_trails=1, diffusion_step=args.diffusion_step, required_sync=False)
            end_event.record()
        if required_sync:
            stream.synchronize()
            duration = time.time() - req_start
            print("Request duration: {:.3f} ms - {}".format(duration*1000, self.idx))
            durations.append(time.time() - req_start)


    def run_parallel_req_v2(self, num_trails):

        start = time.time()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trails)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trails)]
        for i in range(num_trails):
            worker_num = 18
            stream_id = i%worker_num
            time.sleep(args.req_interval)

            with torch.cuda.stream(self.streams[stream_id]):
                start_events[i].record()
                self.run_V_cuda_graphs(num_trails=1, required_sync=False)
                self.run_L_cuda_graphs(num_trails=1, diffusion_step=args.diffusion_step//args.diffusion_stage_num, required_sync=False)
                end_events[i].record()

        torch.cuda.synchronize()
        total_duration = time.time() - start
        durations = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        return durations, total_duration


    def run_parallel_req(self, num_trails):

        start = []
        durations = []
        threads = []
        for i in range(num_trails):
            thread = threading.Thread(target=self.run_single_request, args=(durations, i))
            threads.append(thread)

        start = time.time()
        for thread in threads:
            time.sleep(args.req_interval)
            thread.start()

        for thread in threads:
            thread.join()
        
        torch.cuda.synchronize()
        total_duration = time.time() - start

        return durations, total_duration


    def run_ts(self, task_plan):
        if args.profile_mode == 'base':
            stream_id = 0
            for stage in task_plan:
                if stage == 'e':
                    with torch.cuda.stream(self.streams[stream_id]):
                        self.graphs['encode'].replay()
                elif stage == 'p':
                    with torch.cuda.stream(self.streams[stream_id]):
                        self.graphs['prefill'].replay()
                elif stage == 'd':
                    with torch.cuda.stream(self.streams[stream_id]):
                        self.graphs['decode'].replay()
                
                stream_id = stream_id + 1

        elif args.profile_mode == 'flashinfer':

            if 'd' in task_plan:
                with torch.cuda.stream(self.streams[0]):
                    self.graphs['batch_decode'].replay()
            for i in range(task_plan.count('p')):
                with torch.cuda.stream(self.streams[i+2]):
                    self.graphs['prefill'].replay()
            if 'e' in task_plan:
                with torch.cuda.stream(self.streams[1]):
                    self.graphs['encode'].replay()


    def run_benchmarks(self,
                       mode: str,
                       use_cuda_graphs: bool,
                       num_trails: int,
                       sche_plan,
                       ):
        if mode == 'seq' and use_cuda_graphs:
            durations = []
            for i in range(num_trails):
                self.run_single_request(durations, 0)
            torch.cuda.synchronize()
            print("Query duration: {:.2f} ms".format(np.mean(durations)*1000))

        elif mode == 'pipe':
            start = time.time()
            for i in range(num_trails):
                self.run_VL_ms()
                torch.cuda.synchronize()
            duration = time.time() - start
            print("Query duration: {:.2f} ms".format(duration/num_trails*2*1000))

        elif mode == 'parallel':
            durations, total_duration = self.run_parallel_req(num_trails=num_trails)
            print("Query latency: {:.2f} ms".format(np.mean(durations)*1000))
            print("IN throughput: {:.2f}".format(1/args.req_interval))
            print("OUT throughput: {:.2f}".format(num_trails/total_duration))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))

        elif mode == 'parallel_v2':
            durations, total_duration = self.run_parallel_req_v2(num_trails=num_trails)
            print("Query latency: {:.2f} ms".format(np.mean(durations)))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))

            
        elif mode == 'ours':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for i in range(args.trail_num + args.warmup_num):
                if i == args.warmup_num:
                    start_time = time.time()

                if i == args.warmup_num:
                    start_event.record()

                # First execute the encoder (resnet18), then synchronize,
                # then run multiple diffusion iterations in parallel
                with torch.cuda.stream(self.streams[0]):
                    self.graphs['resnet'].replay()
                torch.cuda.synchronize()
                
                backbone_streams = []
                # # graph-first graph launching
                # for j, graph_name in enumerate(self.graphs['backbone']):
                #     stream = self.streams[j+1]
                #     with torch.cuda.stream(stream):
                #         for diffusion_step in range(args.diffusion_step // args.diffusion_stage_num):
                #             self.graphs['backbone'][j].replay()
                #     backbone_streams.append(stream)

                # iteration-first graph launching
                for diffusion_step in range(args.diffusion_step // args.diffusion_stage_num):
                    for j, graph_name in enumerate(self.graphs['backbone']):
                        stream = self.streams[j+1]
                        with torch.cuda.stream(stream):
                            self.graphs['backbone'][j].replay()
                    backbone_streams.append(stream)                

                # Synchronize all backbone streams before recording end_event
                for stream in backbone_streams:
                    stream.synchronize()

                if i == args.warmup_num:
                    end_event.record()
                torch.cuda.synchronize()

                if i == args.warmup_num:
                    duration = start_event.elapsed_time(end_event)
                    print("Duration of graphs: ", duration)

            frame_interval = (time.time() - start_time) / args.trail_num
            print("Frame interval: {:.4f} s".format(frame_interval))
            print("Throughput: {:.2f}".format(1/frame_interval))

        elif mode == 'profile':
            raise NotImplementedError


def diffusion_run(sche_plan=None, mode='profile', model_type='transformer'):
    print("Start DiffusionPolicy inference...")
    mp.set_start_method('spawn')

    res = None
    profiler.start()

    if mode == 'profile':
        e = Diffusion_engine(model_type=model_type)
        res = e.run_benchmarks(mode='profile',
                               use_cuda_graphs=True,
                               num_trails=100,
                               sche_plan=sche_plan)
    elif mode == 'parallel_sp':
        worker_num = args.worker_num
        e = [Diffusion_engine(idx=i, model_type=model_type) for i in range(worker_num)]
        start = time.time()

        streams = [torch.cuda.Stream() for i in range(worker_num)]
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_trails)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_trails)]

        for i in range(args.num_trails):
            worker_id = i%worker_num
            time.sleep(args.req_interval)

            e[worker_id].run_single_request_v2(streams[worker_id], start_events[i], end_events[i])

        torch.cuda.synchronize()
        total_duration = time.time() - start
        durations = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        print("GPU latency: {:.2f} ms".format(np.mean(durations)))

        # Below four lines is an estimation
        start_times = [i * args.req_interval * 1000 for i in range(args.num_trails)]
        gpu_time = total_duration * 1000 / args.num_trails * worker_num
        end_times = sorted([gpu_time + i * gpu_time for i in range(math.ceil(args.num_trails/worker_num))] * worker_num)[:args.num_trails]
        e2e_durations = [end_times[i] - start_times[i] for i in range(args.num_trails)]
        
        print("E2E latency: {:.2f} ms".format(np.mean(e2e_durations)))
        print("IN throughput: {:.2f}".format(1/args.req_interval))
        print("OUT throughput: {:.2f}".format(args.num_trails/total_duration))

    else:
        e = Diffusion_engine(model_type=model_type)
        e.run_benchmarks(mode=mode,
                         use_cuda_graphs=True,
                         num_trails=100,
                         sche_plan=sche_plan)

    torch.cuda.synchronize()
    profiler.stop()

    print("DiffusionPolicy inference finished.")
    return res


if __name__ == "__main__":
    diffusion_run()
    exit()

