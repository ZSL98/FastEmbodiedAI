import torch
import numpy as np

import os
import sys
import time
import threading
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Value, Array

import torch.cuda.profiler as profiler
from cuda import cuda, cudart

from .encoder.model import vision_transformer
from .llm.model import llama

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args

class LLaVa_engine:
    # [TODO]: The projection and tokenizers are omitted in this implementation!

    def __init__(self):
        self.text_max_seq_len = 256

        # prepare caches for tensors
        text = torch.randint(0, 256, (1, 128+576 )).to("cuda")
        img = torch.randn(1, 3, 336, 336).to("cuda")
        context = torch.randn(1, 64, 512).to("cuda")
        single_token = torch.randint(0, 256, (1, 1)).to("cuda")
        self.caches = { 'text': text,
                        'img': img,
                        'context': context,
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

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(36)]

    def generate_cuda_graphs(self):
        recording_kwargs = {}

        ## Make cuda graph for the prefill phase
        # [BUG]: I have to run the following command once to make the cuda graph generated properly
        # [FIXME]: The output caches of the graphs have not been designed yet
        # [FIXME]: The decode phase is static, which is just an approximate
        out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                            seq_len = self.text_max_seq_len,
                                                            kv_cache = None)
        with torch.cuda.graph(self.graphs['prefill'], **recording_kwargs):
            out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                                seq_len = self.text_max_seq_len, 
                                                                kv_cache = None)
        self.graphs['prefill'].replay()
        torch.cuda.synchronize()
        self.kv_cache = new_cache
        # print("self.kv_cache shape: ", self.kv_cache.attn_intermediates[0].cached_kv[0].shape)
        print("====== Graph for prefill generated ======")

        ## Make cuda graph for the decode phase
        out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['single_token'], 
                                                            seq_len = self.text_max_seq_len, 
                                                            kv_cache = self.kv_cache)
        with torch.cuda.graph(self.graphs['decode'], **recording_kwargs):
            out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['single_token'], 
                                                                seq_len = self.text_max_seq_len, 
                                                                kv_cache = self.kv_cache)
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

        elif mode == 'profile':
            pass



def llava_run(sche_plan=None, mode='profile'):
    print("Start LLaVa inference...")
    mp.set_start_method('spawn')

    e = LLaVa_engine()
    res = None
    profiler.start()
    if sche_plan is None:
        e.run_benchmarks(mode=mode,
                         use_cuda_graphs=True,
                         num_trails=100,
                         sche_plan=None)
    else:
        res = e.run_benchmarks(mode='profile',
                               use_cuda_graphs=True,
                               num_trails=1000,
                               sche_plan=sche_plan)
    torch.cuda.synchronize()
    profiler.stop()

    print("Finished.")
    return res


if __name__ == "__main__":
    llava_run()
    exit()


