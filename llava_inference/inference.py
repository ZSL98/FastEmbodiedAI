import torch
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

from .encoder.model import vision_transformer
from .llm.model import llama

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args, pattern_analyze


class LLaVa_engine:
    # [TODO]: The projection and tokenizers are omitted in this implementation!

    def __init__(self, idx = 0):
        self.idx = idx
        self.text_max_seq_len = 256
        self.input_seq_len = args.input_seq_len + 576

        # prepare caches for tensors
        text = torch.randint(0, 256, (1, self.input_seq_len)).to("cuda")
        img = torch.randn(1, 3, 336, 336).to("cuda")
        single_token = torch.randint(0, 256, (1, 1)).to("cuda")
        self.caches = { 'text': text,
                        'img': img,
                        'single_token': single_token}

        # prepare models
        vit = vision_transformer().to("cuda")
        llm = llama().to("cuda")
        self.models = {'vit': vit, 
                       'llm': llm}

        # prepare cuda graphs
        self.graphs = {'encode': torch.cuda.CUDAGraph(),
                        'prefill': torch.cuda.CUDAGraph(),
                        'decode': torch.cuda.CUDAGraph()}
        self.generate_cuda_graphs()
        self.ours_graphs = {}

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(36)]

        max_batch_size = 8
        self.graphs['batch_decode'] = torch.cuda.CUDAGraph()
        batch_single_token = torch.randint(0, 256, (max_batch_size, 1)).to("cuda")
        self.caches['batch_single_token'] = batch_single_token
        self.k_cache_trans = torch.randn(max_batch_size, 32, self.input_seq_len, 128).half().to("cuda")
        self.v_cache_trans = torch.randn(max_batch_size, 32, self.input_seq_len, 128).half().to("cuda")
        self.trans_cache = [self.k_cache_trans, self.v_cache_trans]



    def generate_cuda_graphs(self):
        recording_kwargs = {}

        ## Make cuda graph for the prefill phase
        # [BUG]: I have to run the following command once to make the cuda graph generated properly
        # [FIXME]: The output caches of the graphs have not been designed yet
        # [FIXME]: The decode phase is static, which is just an approximate
        out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                            seq_len = self.text_max_seq_len,
                                                            kv_cache = None)
        del out
        del new_cache
        with torch.cuda.graph(self.graphs['prefill'], **recording_kwargs):
            self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                                seq_len = self.text_max_seq_len, 
                                                                kv_cache = None)
        self.graphs['prefill'].replay()
        torch.cuda.synchronize()
        # self.kv_cache = self.new_cache
        # print("self.kv_cache shape: ", self.kv_cache.attn_intermediates[0].cached_kv[0].shape)
        print("====== Graph for prefill generated ======")

        ## Make cuda graph for the decode phase
        out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['single_token'], 
                                                            seq_len = self.text_max_seq_len, 
                                                            kv_cache = self.new_cache)
        del out
        del new_cache
        with torch.cuda.graph(self.graphs['decode'], **recording_kwargs):
            self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['single_token'], 
                                                                seq_len = self.text_max_seq_len, 
                                                                kv_cache = self.new_cache)
        self.graphs['decode'].replay()
        torch.cuda.synchronize()
        print("====== Graph for decode generated ======")

        ## Make cuda graph for the vision encoder
        with torch.cuda.graph(self.graphs['encode'], **recording_kwargs):
            out = self.models['vit'](self.caches['img'])
            # print("out shape: ", out.shape)
        self.graphs['encode'].replay()
        torch.cuda.synchronize()
        print("====== Graph for vision generated ======")


    def generate_graph_group(self, ts_epd_num):
        graph_group = {}
        ts_encode_num, ts_prefill_num, ts_decode_num = ts_epd_num

        recording_kwargs = {}
        if ts_decode_num != 0:
            new_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(new_graph, **recording_kwargs):
                self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(
                    self.caches['batch_single_token'][:ts_decode_num, ...], 
                    seq_len = self.text_max_seq_len,
                    kv_cache = None,
                    supercache = [self.trans_cache[0][:ts_decode_num, ...], self.trans_cache[1][:ts_decode_num, ...]])

            new_graph.replay()
            graph_name = str(ts_decode_num) + 'd'
            print(graph_name)
            self.ours_graphs[graph_name] = new_graph
            
            torch.cuda.synchronize()
            graph_group[graph_name] = self.ours_graphs[graph_name]
        
        if ts_prefill_num != 0:
            for i in range(ts_prefill_num):
                graph_group['p'] = self.graphs['prefill']

        if ts_encode_num != 0:
            for i in range(ts_encode_num):
                graph_group['e'] = self.graphs['encode']

        print('done')

        return graph_group


    def generate_extra_cuda_graphs(self, ts_encode_num, ts_prefill_num, ts_decode_num, graph_name):

        # # ts_decode_num = 1
        # batch_single_token = torch.randint(0, 256, (ts_decode_num, 1)).to("cuda")
        # self.caches['batch_single_token'] = batch_single_token

        # self.k_cache_trans = torch.randn(ts_decode_num, 32, self.input_seq_len, 128).half().to("cuda")
        # # self.v_cache_trans = torch.randn(ts_decode_num, 32, self.input_seq_len, 128).half().to("cuda")
        # self.trans_cache = [self.k_cache_trans, self.k_cache_trans]

        recording_kwargs = {}
        if ts_decode_num != 0:
            new_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(new_graph, **recording_kwargs):
                self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(
                    self.caches['batch_single_token'][:ts_decode_num, ...], 
                    seq_len = self.text_max_seq_len,
                    kv_cache = None,
                    supercache = [self.trans_cache[0][:ts_decode_num, ...], self.trans_cache[1][:ts_decode_num, ...]])

            new_graph.replay()
            self.graphs[graph_name] = new_graph
            torch.cuda.synchronize()


    def run_cuda_graphs(self, num_trails):
        for i in range(num_trails):
            self.graphs['encode'].replay()
            self.graphs['prefill'].replay()
            self.graphs['decode'].replay()
        torch.cuda.synchronize()


    def run_V_cuda_graphs(self, num_trails=1, required_sync=True):
        for i in range(num_trails):
            self.graphs['encode'].replay()
            if required_sync:
                torch.cuda.synchronize()


    def run_L_cuda_graphs(self, num_trails=1, out_seq_len=64, required_sync=True):
        for i in range(num_trails):
            self.graphs['prefill'].replay()
            for token in range(out_seq_len-1):
                self.graphs['decode'].replay()
                if required_sync:
                    torch.cuda.synchronize()


    def run_VL_ms(self):
        with torch.cuda.stream(self.streams[0]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
        with torch.cuda.stream(self.streams[1]):
            self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)


    def run_basic(self, num_trails):
        pass


    def run_single_request(self, durations, i):
        worker_num = 2
        stream_id = i%worker_num
        req_start = time.time()

        with torch.cuda.stream(self.streams[stream_id]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
            self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)
        
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
            self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)
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
                self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)
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
            print("Prepare required tensors and cuda graphs.")
            all_graph_group = []
            sche_duration = {}
            for task_plan, count in sche_plan.items():
                if count != 1:
                    ts_epd_num = pattern_analyze(task_plan)
                    print("ts_epd_num: ", ts_epd_num)
                    graph_group = self.generate_graph_group(ts_epd_num)
                    all_graph_group.append(graph_group)
            
            for i in range(args.trail_num + args.warmup_num):
                if i == args.warmup_num:
                    start_time = time.time()

                for group in all_graph_group:
                    for j, graph_name in enumerate(group):
                        # print("execute: ", graph_name)
                        with torch.cuda.stream(self.streams[j]):
                            group[graph_name].replay()
                    torch.cuda.synchronize()
            
            frame_interval = (time.time() - start_time) / args.trail_num
            print("Total duration: {:.4f} s".format(frame_interval))
            print("Throughput: {:.2f}".format(1/frame_interval))
            

        elif mode == 'profile':
            if args.profile_mode == 'flashinfer':
                del self.graphs['decode']
            sche_duration = {}

            for task_plan in sche_plan:
                # prepare batch graph
                ts_encode_num, ts_prefill_num, ts_decode_num = pattern_analyze(task_plan)
                alloc1 = torch.cuda.memory_allocated()
                self.generate_extra_cuda_graphs(ts_encode_num, ts_prefill_num, ts_decode_num, 'batch_decode')
                alloc2 = torch.cuda.memory_allocated()
                print('Generated CUDA graph size: ', (alloc2 - alloc1)/(1024*1024))

                for i in range(args.warmup_num):
                    self.run_ts(task_plan)
                torch.cuda.synchronize()
                # Time the ts duration
                start_time = time.time()
                for i in range(args.trail_num):
                    self.run_ts(task_plan)
                    torch.cuda.synchronize()

                task_duration = (time.time() - start_time) * 1000 / args.trail_num
                sche_duration[task_plan] = task_duration
                print(task_plan, task_duration)

                if ts_decode_num != 0:
                    del self.graphs['batch_decode']
                    # del self.graphs['decode']
                    # del self.k_cache_trans
                    # del self.v_cache_trans
                    # del self.trans_cache
                    del self.new_cache
                    del self.out
                    torch.cuda.empty_cache()

            return sche_duration


def llava_run(sche_plan=None, mode='profile'):
    print("Start LLaVa inference...")
    mp.set_start_method('spawn')

    res = None
    profiler.start()

    if mode == 'profile':
        e = LLaVa_engine()
        res = e.run_benchmarks(mode='profile',
                               use_cuda_graphs=True,
                               num_trails=100,
                               sche_plan=sche_plan)
    elif mode == 'parallel_sp':
        worker_num = args.worker_num
        e = [LLaVa_engine(idx=i) for i in range(worker_num)]
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
        e = LLaVa_engine()
        e.run_benchmarks(mode=mode,
                         use_cuda_graphs=True,
                         num_trails=100,
                         sche_plan=sche_plan)

    torch.cuda.synchronize()
    profiler.stop()

    print("LLaVa inference finished.")
    return res


if __name__ == "__main__":
    llava_run()
    exit()


