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
from cuda import cuda, cudart

from .encoder.model import vision_transformer, vit_sliced
from .llm.model import llama, llama_sliced

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args, pattern_analyze, pattern_analyze_ad

from x_transformers import LayerIntermediates
from x_transformers.attend import Intermediates

class LLaVa_sliced_engine:
    def __init__(self, sche_plan):
        self.text_max_seq_len = 256
        self.input_seq_len = args.input_seq_len + 576

        self.sche_plan = sche_plan
        # self.task_plan = task_plan
        self.models = {}
        self.generate_models()
        self.graphs = {}
        self.all_graph_group = []

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(36)]

    def generate_models(self):
        vit = vision_transformer().to("cuda")
        llm = llama().to("cuda")
        self.models = {'vit': vit, 
                       'llm': llm}


    def generate_graph_group(self, ts_detail):
        graph_group = {}
        ts_encode_list, ts_prefill_list, ts_decode_list = [], [], []
        
        for item in ts_detail:
            if list(item.keys())[0] == 'e':
                ts_encode_list.append(list(item.values())[0])
            elif list(item.keys())[0] == 'p':
                ts_prefill_list.append(list(item.values())[0])
            elif list(item.keys())[0] == 'd':
                ts_decode_list.append(list(item.values())[0])

        # The below arrangement is specific to different model
        recording_kwargs = {}
        if len(ts_decode_list) != 0:
            bs = len(ts_decode_list)
            # print("len(ts_decode_list): ", bs)
            self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(
                graph_input = self.caches['batch_single_token'][:bs, ...],
                seq_len = self.text_max_seq_len,
                kv_cache = self.caches['kv_cache_bs' + str(bs)],
                slice_num = args.decode_n,
                slice_id = 0,
                pre_compute = True,
                post_compute = True,
            )
            # print("self.out, self.new_cache: ", self.out.shape, self.new_cache.attn_intermediates[1].cached_kv[0].shape)
            new_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(new_graph, **recording_kwargs):
                self.out, self.new_cache = self.models['llm'].wrapped_decoder.make_graph(
                    self.caches['batch_single_token'][:bs, ...],
                    seq_len = self.text_max_seq_len,
                    kv_cache = self.caches['kv_cache_bs' + str(bs)],
                    slice_num = args.decode_n,
                    slice_id = 0,
                    pre_compute = True,
                    post_compute = True,
                )
        
            new_graph.replay()
            graph_group['d'] = new_graph
        
        self.prefill_cache = {}
        self.new_cache_tmp = {}
        if len(ts_prefill_list) != 0:
            for i in range(len(ts_prefill_list)):
                # print("ts_prefill_list[i]: ", ts_prefill_list[i])
                self.prefill_cache[i], self.new_cache_tmp[i] = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'][i], 
                                                                        seq_len = 256, 
                                                                        slice_num = args.prefill_n,
                                                                        slice_id = ts_prefill_list[i],
                                                                        kv_cache = None)
                graph_group['p' + str(ts_prefill_list[i])] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_group['p' + str(ts_prefill_list[i])], stream=self.streams[i+1]):
                    # TODO: This is not correct
                    self.prefill_cache[i], self.new_cache_tmp[i] = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'][i], 
                                                                        seq_len = 256, 
                                                                        slice_num = args.prefill_n,
                                                                        slice_id = ts_prefill_list[i],
                                                                        kv_cache = None)
                graph_group['p' + str(ts_prefill_list[i])].replay()                                       
                                                            
        if len(ts_encode_list) != 0:
            for i in range(len(ts_encode_list)):
                out = self.models['vit'](self.caches['img'], slice_num = args.encoder_n, slice_id = ts_encode_list[i])
                graph_group['e' + str(ts_encode_list[i])] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_group['e' + str(ts_encode_list[i])] , **recording_kwargs):
                    out = self.models['vit'](self.caches['img'], slice_num = args.encoder_n, slice_id = ts_encode_list[i])
                graph_group['e' + str(ts_encode_list[i])].replay()

        return graph_group


    def manage_cache(self, all_ts_detail):
        self.caches = {}
        self.caches['tokens'] = []
        self.caches['encoder_related'] = []
        self.caches['img'] = torch.randn(1, 3, 336, 336).to("cuda")
        self.caches['text'] = [torch.randint(0, 256, (1, self.input_seq_len)).to("cuda") for i in range(16)]

        max_active_req = 1
        all_kv_cache_requirement = []

        for ts_detail in all_ts_detail:
            ts_encode_num, ts_prefill_num, ts_decode_num = 0, 0, 0
            for item in ts_detail:
                if list(item.keys())[0] == 'e':
                    ts_encode_num = ts_encode_num + 1
                elif list(item.keys())[0] == 'p':
                    ts_prefill_num = ts_prefill_num + 1
                elif list(item.keys())[0] == 'd':
                    ts_decode_num = ts_decode_num + 1
                active_req = ts_decode_num + ts_prefill_num
                if active_req > max_active_req:
                    max_active_req = active_req

                all_kv_cache_requirement.append(ts_prefill_num)
                all_kv_cache_requirement.append(ts_decode_num)

        all_kv_cache_requirement = list(set(all_kv_cache_requirement))

        print("all_kv_cache_requirement: ", all_kv_cache_requirement)

        decoder_layer_num = 4
        k_cache = [torch.randn(max_active_req, 32, self.input_seq_len, 128).half().to("cuda") for i in range(decoder_layer_num)] #16 is the number of layers
        v_cache = [torch.randn(max_active_req, 32, self.input_seq_len, 128).half().to("cuda") for i in range(decoder_layer_num)]

        for bs in all_kv_cache_requirement:
            kv_cache = LayerIntermediates()
            kv_cache.attn_intermediates = [Intermediates()] * decoder_layer_num  #16 is the number of layers
            for layer in range(decoder_layer_num):
                kv_cache.attn_intermediates[layer].cached_kv = (k_cache[layer][:bs, ...], v_cache[layer][:bs, ...])
            self.caches['kv_cache_bs' + str(bs)] = kv_cache
    
        self.caches['batch_single_token'] = torch.randint(0, 256, (max_active_req, 1)).to("cuda")


    def init_benchmark(self):
        all_ts_detail = []
        for task_plan, count in self.sche_plan.items():
            if count != 1:
                # JOB_PLAN: e, e, e, p, p, d, d, d, d, d, d
                # job_plan: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10
                # job_plan:             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10
                # job_plan:                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10
                # job_plan:                                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10

                # task_plan: (7, 3)     --> ts_detail: [{'d': 0}, {'p': 0}]
                # task_plan: (8, 4, 0)  --> ts_detail: [{'d': 0}, {'p': 1}, {'e': 0}]
                # task_plan: (9, 5, 1)  --> ts_detail: [{'d': 0}, {'d': 0}, {'e': 1}]
                # task_plan: (10, 6, 2) --> ts_detail: [{'d': 0}, {'d': 0}, {'e': 2}]

                ts_detail = pattern_analyze_ad(task_plan)
                print("ts_detail: \n", ts_detail)
                all_ts_detail.append(ts_detail)
        
        self.manage_cache(all_ts_detail)
        for ts_detail in all_ts_detail:
            graph_group = self.generate_graph_group(ts_detail)
            self.all_graph_group.append(graph_group)

    def run_benchmark(self):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        for i in range(args.trail_num + args.warmup_num):
            if i == args.warmup_num:
                start_time = time.time()

            for group in self.all_graph_group:
                for j, graph_name in enumerate(group):
                    # if i == 0:
                    with torch.cuda.stream(self.streams[j]):
                        if i == args.warmup_num:
                            start_events[j].record()
                        group[graph_name].replay()
                        if i == args.warmup_num:
                            end_events[j].record()
                # for j, graph_name in enumerate(group):
                #     self.streams[j].synchronize()
                #     print("Execute: ", j, graph_name)
                torch.cuda.synchronize()

                if i == args.warmup_num:
                    duration = [s.elapsed_time(e) for s, e in zip(start_events[:j+1], end_events[:j+1])]
                    print("Duration of graphs: ", duration)

        frame_interval = (time.time() - start_time) / args.trail_num
        print("Frame interval: {:.4f} s".format(frame_interval))
        print("Throughput: {:.2f}".format(1/frame_interval))


def llava_run_sliced(sche_plan):
    profiler.start()
    e = LLaVa_sliced_engine(sche_plan)
    e.init_benchmark()
    e.run_benchmark()

    torch.cuda.synchronize()
    profiler.stop()

    print("LLaVa sliced inference finished.")
