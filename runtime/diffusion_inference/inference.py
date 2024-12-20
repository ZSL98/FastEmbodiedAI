import torch
import torchvision.models as models
import numpy as np

import os
import sys
import time
import math
import threading
import asyncio
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Value, Array

import torch.cuda.profiler as profiler
# from cuda import cuda, cudart

from .transformer.model import DiffusionTransformer
from .cnn.model import DiffusionCNN
from diffusion_policy.common.pytorch_util import split_with_skew, split_with_skew_no_acc

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args, pattern_analyze
from contextlib import nullcontext


class Diffusion_engine:
    # [TODO]: The projection and tokenizers are omitted in this implementation!

    # The configurations of the Diffusion Policy (Transformer-based) [2024.11.25]:

    def __init__(self, 
                 idx = 0,
                 model_type='transformer'):
        self.idx = idx

        args.diffusion_step = int(args.generation_scale * args.diffusion_step)
        # prepare caches for tensors
        if args.mode == 'seq':
            args.diffusion_stage_num = 1

        self.n_replica = args.diffusion_stage_num
        if args.mode == 'parallel_v2':
            self.n_replica = args.worker_num
        
        obs_num = 2
        # assert args.input_img_num*obs_num*args.perception_scale%1 == 0, "DiffusionPolicy Perception scaling warning"
        img = [torch.randn(int(args.input_img_num*obs_num*args.perception_scale), 3, args.input_img_size, args.input_img_size).to(torch.bfloat16).to("cuda") for i in range(self.n_replica)]
        if model_type == 'transformer':
            trajectory = [torch.randn(1, args.input_traj_transformer_size, args.input_dim).to(torch.bfloat16).to("cuda") for i in range(self.n_replica)]
            cond = [torch.randn(1, obs_num, args.cond_dim).to(torch.bfloat16).to("cuda") for i in range(self.n_replica)]
        elif model_type == 'cnn':
            trajectory = [torch.randn(1, args.input_traj_cnn_size, args.input_dim).to(torch.bfloat16).to("cuda") for i in range(self.n_replica)]
            cond = [torch.randn(1, obs_num*args.cond_dim).to(torch.bfloat16).to("cuda") for i in range(self.n_replica)]     
        timestep = [torch.tensor(0, dtype=torch.long, device="cuda") for i in range(self.n_replica)]
        self.caches = { 'img': img,
                        'trajectory': trajectory,
                        'timestep': timestep,
                        'cond': cond}

        # prepare models
        resnet = models.resnet18().to(torch.bfloat16).to("cuda")
        if model_type == 'transformer':
            backbone = DiffusionTransformer(
                input_dim=args.input_dim,
                output_dim=args.input_dim,
                cond_dim=args.cond_dim,
                n_emb=args.n_emb
            ).to(torch.bfloat16).to("cuda")
        elif model_type == 'cnn':
            backbone = DiffusionCNN(
                input_dim=args.input_dim,
                global_cond_dim=args.global_cond_dim
            ).to(torch.bfloat16).to("cuda")
        
        print("Perception params: %e" % sum(p.numel() for p in resnet.parameters()))
        print("Generation params: %e" % sum(p.numel() for p in backbone.parameters()))

        self.models = {'resnet': resnet,
                       'backbone': backbone}

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(36)]

        # prepare cuda graphs
        self.graphs = {'resnet': [torch.cuda.CUDAGraph() for i in range(self.n_replica)],
                        'backbone': [torch.cuda.CUDAGraph() for i in range(self.n_replica)]}
        self.generate_cuda_graphs()
        self.ours_graphs = {}


    def generate_cuda_graphs(self):

        ## Make cuda graph for the vision encoder
        self.resnet_out = self.models['resnet'](self.caches['img'][0])
        self.resnet_out = {}
        for graph_id in range(self.n_replica):
            with torch.cuda.graph(self.graphs['resnet'][graph_id], stream=self.streams[graph_id]):
                self.resnet_out = self.models['resnet'](self.caches['img'][graph_id])
        torch.cuda.synchronize()
        print("====== Graph for resnet generated ======")

        self.backbone_out = self.models['backbone'](self.caches['trajectory'][0], self.caches['timestep'][0], self.caches['cond'][0])
        self.backbone_out = {}
        for graph_id in range(self.n_replica):
            with torch.cuda.graph(self.graphs['backbone'][graph_id], stream=self.streams[self.n_replica+graph_id]):
                self.backbone_out[graph_id] = self.models['backbone'](self.caches['trajectory'][graph_id], 
                                                                            self.caches['timestep'][graph_id], 
                                                                            self.caches['cond'][graph_id])
            
        torch.cuda.synchronize()
        print("====== Graph for backbone generated ======")


    def run_cuda_graphs(self, num_trails):
        for i in range(num_trails):
            self.graphs['resnet'][0].replay()
            for diffusion_step in range(args.diffusion_step):
                self.graphs['backbone'][0].replay()
        torch.cuda.synchronize()


    def run_V_cuda_graphs(self, num_trails=1, required_sync=True, 
                          graph_id=0, stream=None, start_event=None, end_event=None):
        
        context = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with context:
            if start_event is not None:
                start_event.record(stream)
            for i in range(num_trails):
                self.graphs['resnet'][graph_id].replay()
                if required_sync:
                    torch.cuda.synchronize()
            if end_event is not None:
                end_event.record(stream)


    def run_L_cuda_graphs(self, num_trails=1, diffusion_step=64, required_sync=True, 
                          graph_id=0, stream=None, start_event=None, end_event=None, slice_list=None):
        if not isinstance(stream, list):
            context = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with context:
                if start_event is not None:
                    start_event.record(stream)
                for i in range(num_trails):
                    for step in range(diffusion_step):
                        self.graphs['backbone'][graph_id].replay()
                        if required_sync:
                            torch.cuda.synchronize()
                if end_event is not None:
                    end_event.record(stream)
        else:
            assert slice_list is not None, "slice_list should not be none"
            if start_event is not None:
                start_event.record()
            for i in range(num_trails):
                for id, ths_stream in enumerate(stream):
                    with torch.cuda.stream(ths_stream):
                        for _ in range(slice_list[id]):
                            self.graphs['backbone'][id].replay()
            if end_event is not None:
                end_event.record()
                    

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
        # print("Request duration: {:.3f} ms".format(duration*1000))
        durations.append(time.time() - req_start)


    def run_single_request_v2(self, stream, start_event, end_event, required_sync=False, graph_id=0):
        req_start = time.time()
        # print("Launch - {}".format(self.idx))
        
        with torch.cuda.stream(stream):
            start_event.record()
            self.run_V_cuda_graphs(num_trails=1, required_sync=False, graph_id=graph_id)
            self.run_L_cuda_graphs(num_trails=1, diffusion_step=args.diffusion_step, required_sync=False, graph_id=graph_id)
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
        threads = []
        for i in range(num_trails):
            graph_id = i%args.worker_num
            t = threading.Timer(i * args.req_interval, 
                                self.run_single_request_v2, 
                                [self.streams[graph_id], start_events[i], end_events[i], False, graph_id])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        torch.cuda.synchronize()
        total_duration = time.time() - start
        durations = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        return durations, total_duration

    def run_VL_decouple(self, num_trails, parallel_L=False):

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start = time.time()
        scale = 25
        thread_V = threading.Thread(target=self.run_V_cuda_graphs, args=(num_trails*scale, 
                                                                        False, 0, 
                                                                        self.streams[0], 
                                                                        start_events[0],
                                                                        end_events[0]))
        if parallel_L == False:
            # Prefill and decode are sequential and then parallelized with V
            thread_L = threading.Thread(target=self.run_L_cuda_graphs, args=(num_trails, 
                                                                            args.diffusion_step, 
                                                                            False, 0,
                                                                            self.streams[1],
                                                                            start_events[1],
                                                                            end_events[1]))
        else:
            # Only a prefill is parallelized with V
            slice_list = split_with_skew_no_acc(args.diffusion_step, args.diffusion_stage_num, args.diffusion_slice_skewness)
            print("slice_list: ", slice_list)
            thread_L = threading.Thread(target=self.run_L_cuda_graphs, args=(num_trails,
                                                                            args.diffusion_step, 
                                                                            False, 0,
                                                                            self.streams[1:len(slice_list)+1],
                                                                            start_events[1],
                                                                            end_events[1],
                                                                            slice_list))

        thread_V.start()
        thread_L.start()

        thread_V.join()
        thread_L.join()

        torch.cuda.synchronize()
        total_duration = time.time() - start

        durations = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        print(durations)
        assert durations[0] > durations[1], "V is finished before L, adjust the scale in run_V_cuda_graphs()"

        return durations[1]/1000

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
            print("Throughput: {:.3f}".format(1/np.mean(durations)))

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
            print("Query latency: {:.3f} ms".format(np.mean(durations)))
            print("Query duration: {:.3f}".format(total_duration*1000/num_trails))
            print("Throughput: {:.3f}".format(1/(total_duration/num_trails)))

        elif mode == 'decouple':
            total_duration = self.run_VL_decouple(num_trails=num_trails, parallel_L=False)
            print("OUT throughput: {:.2f}".format(num_trails/total_duration))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))

        elif mode == 'ours_decouple':
            total_duration = self.run_VL_decouple(num_trails=num_trails, parallel_L=True)
            print("OUT throughput: {:.2f}".format(num_trails/total_duration))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))
            
        elif mode == 'ours':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            slice_list = split_with_skew_no_acc(args.diffusion_step, args.diffusion_stage_num, args.diffusion_slice_skewness)
            for i in range(args.trail_num + args.warmup_num):
                if i == args.warmup_num:
                    start_time = time.time()

                if i == args.warmup_num:
                    start_event.record()

                # First execute the encoder (resnet18), then synchronize,
                # then run multiple diffusion iterations in parallel
                with torch.cuda.stream(self.streams[0]):
                    self.graphs['resnet'][0].replay()
                torch.cuda.synchronize()
                
                backbone_streams = []
                # print("slice_list: ", slice_list)
                # graph-first graph launching
                for j, graph_name in enumerate(self.graphs['backbone']):
                    stream = self.streams[j+1]
                    with torch.cuda.stream(stream):
                        # for diffusion_step in range(args.diffusion_step // args.diffusion_stage_num):
                        for diffusion_step in range(slice_list[j]):
                            self.graphs['backbone'][j].replay()
                    backbone_streams.append(stream)

                # # iteration-first graph launching
                # for diffusion_step in range(args.diffusion_step // args.diffusion_stage_num):
                #     for j, graph_name in enumerate(self.graphs['backbone']):
                #         stream = self.streams[j+1]
                #         with torch.cuda.stream(stream):
                #             self.graphs['backbone'][j].replay()
                #     backbone_streams.append(stream)                

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
            print("Throughput: {:.3f}".format(1/frame_interval))

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
                         num_trails=20,
                         sche_plan=sche_plan)

    torch.cuda.synchronize()
    profiler.stop()

    print("DiffusionPolicy inference finished.")
    return res


if __name__ == "__main__":
    diffusion_run()
    exit()


