import torch
import torch.nn.functional as F
import time
import math
import torch.multiprocessing as mp
import torch.cuda.profiler as profiler
from cuda import cuda, cudart
from torch.multiprocessing import Process, Value, Array
import os
import numpy as np
import threading

from .encoder0.vision_encoder import vision_encoder
from .encoder0.audio_encoder import audio_encoder
from .encoder1.combiner import combiner
from .llm.encoder import mirasol_encoder
from .llm.decoder import mirasol_decoder

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from sche_plan import args


def audio_process(events, audio_tokens, audio_buffers, buffer_ids, audio_enc):
    # _, audio_cuda_ctx = cuda.cuCtxCreate(0, 0)

    """
    affinity = cuda.CUexecAffinityParam()
    affinity.type = cuda.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT
    affinity.param.smCount.val = 1

    ctx = cuda.cuCtxCreate_v3([affinity], 1, 0, 0)[1]
    cuda.cuInit(0)
    cuda.cuCtxPushCurrent(ctx)

    if torch.cuda.is_available():
        # Perform GPU operations
        pass
    else:
        print("CUDA is not available. Check your GPU configuration.")
    """

    torch.cuda.set_device(1)
    # torch.cuda.init()
    # recording_kwargs = {}
    # with torch.cuda.graph(audio_graph, **recording_kwargs):
    #     encoded_audio = audio_enc(audio_tokens)
    #     _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
    #     encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
    # audio_graph.replay()

    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "2"

    # start_time = time.time()
    for i in range(100):
        # print("audio_process buffer_id: ", buffer_ids[0].value)
        # if isinstance(audio_graph, torch.cuda.CUDAGraph):
        #     print("Is cuda graph.")
        #     audio_graph.replay()

        encoded_audio = audio_enc(audio_tokens)
        _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
        audio_buffers[buffer_ids[0].value] = encoded_audio[:, :1]

        events['af'].set()
        events['cb'].wait()
        # print('combiner_event finished 1')
        # combiner_event.clear()

    """
    current_ctx = cuda.cuCtxGetCurrent()[1]
    print("Current CUDA context ID: ", cuda.cuCtxGetId(current_ctx)[1])
    cudart.cudaDeviceSynchronize()
    """
    value = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    print(f"Child process: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={value}")

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Audio Encoder Elapsed Time: {elapsed_time} seconds")
    
def video_process(events, video_tokens, video_buffers, buffer_ids, vision_enc):
    # start_time = time.time()
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "2"
    for i in range(100):
        # print("video_process buffer_id: ", buffer_ids[0].value)
        encoded_video = vision_enc(video_tokens)
        _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]
        video_buffers[buffer_ids[0].value] = encoded_video[:, :1]

        events['vf'].set()
        events['cb'].wait()
        # print('combiner_event finished 2')
        # combiner_event.clear()

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Video Encoder Elapsed Time: {elapsed_time} seconds")

def combiner_process(events, buffer_ids, audio_buffers, video_buffers, combiner_buffers, combiner_enc):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "2"
    for i in range(100):
        events['af'].wait()
        events['vf'].wait()
        # print('av finished')
        buffer_ids[0].value = 1 - buffer_ids[0].value

        events['af'].clear()
        events['vf'].clear()
        events['cb'].set() # Synchronization is finished
        events['cb'].wait()
        events['cb'].clear()

        audio_and_video_tokens, _ = pack((audio_buffers[1-buffer_ids[0].value], video_buffers[1-buffer_ids[0].value]), 'b n * d')
        audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')
        combined_audio_video_tokens = combiner_enc(audio_and_video_tokens)
        combiner_buffers[buffer_ids[1].value] = combined_audio_video_tokens
        # print("combiner_process buffer_id: ", buffer_ids[1].value)
        events['cf'].set()
        events['eb'].wait()

def encoder_process(events, buffer_ids, combiner_buffers, encoder_buffers, encoder):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "2"
    for i in range(100):
        events['cf'].wait()
        buffer_ids[1].value = 1 - buffer_ids[1].value
        events['cf'].clear()
        events['eb'].set()
        events['eb'].wait()
        events['eb'].clear()
        combined_audio_video_tokens = combiner_buffers[1-buffer_ids[1].value][..., -3:, :]
        combine_ps = [torch.Size([1, 1])]
        combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
        av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
        encoder_buffers[buffer_ids[2].value] = encoder(av_encoder_input)

        events['ef'].set() 
        # The encoder finished its processing. The encoder_buffer is updated and can be used in decoder

        # events['db'].wait()

def decoder_process(events, buffer_ids, encoder_buffers, output, decoder):
    for i in range(100):
        # The decoder does not need to strictly wait for the 'ef' signal.
        # events['ef'].wait()
        buffer_ids[2].value = 1 - buffer_ids[2].value
        events['df'].clear()
        events['eb'].set()
        events['eb'].wait()
        events['eb'].clear()


class Mirasol_engine:

    def __init__(self, DIM=512, idx=0):

        self.idx = idx
        self.text_max_seq_len = 64

        # prepare caches for tensors
        text_tokens = torch.randint(0, 256, (1, 128)).to("cuda")
        audio_tokens = torch.randn(1, 12, DIM).half().to("cuda")
        video_tokens = torch.randn(1, 12, DIM).half().to("cuda")
        av_embeddings = torch.randn(1, 3, DIM).half().to("cuda")
        av_context = torch.randn(1, 18, DIM).half().to("cuda")
        single_token = torch.randint(0, 256, (1, 1)).to("cuda")
        self.caches = {'text': text_tokens, 
                       'single_token': single_token,
                       'audio': audio_tokens, 
                       'video': video_tokens,
                       'av_embeddings': av_embeddings, # This is the context for one frame
                       'av_context': av_context # This is the context for six frames
                       }

        # Those are intermediate caches for small graphs
        encoded_video = torch.randn(1, 1, 4, DIM).half().to("cuda")
        encoded_audio = torch.randn(1, 1, 4, DIM).half().to("cuda")
        combiner_out = torch.randn(1, 8, DIM).half().to("cuda")
        self.extra_caches = {
            'enc_video': encoded_video,
            'enc_audio': encoded_audio,
            'combiner_out': combiner_out,
        }

        # prepare models
        vision_enc = vision_encoder(dim=DIM).half().to("cuda")
        audio_enc = audio_encoder(dim=DIM).half().to("cuda")
        combiner_enc = combiner(dim=DIM).half().to("cuda")
        encoder = mirasol_encoder(dim=DIM).half().to("cuda")
        decoder = mirasol_decoder(dim=DIM).half().to("cuda")
        self.models = {'vision_enc': vision_enc, 
                       'audio_enc': audio_enc, 
                       'combiner_enc': combiner_enc, 
                       'encoder': encoder, 
                       'decoder': decoder}

        self.task_queue = mp.Queue()
        
        serialize_required = False
        if not serialize_required:
            self.engine_init()

    def engine_init(self):
        # prepare cuda graphs
        self.graphs = {'vision': torch.cuda.CUDAGraph(),
                    'audio': torch.cuda.CUDAGraph(),
                    'combiner': torch.cuda.CUDAGraph(),
                    'encoder': torch.cuda.CUDAGraph(),
                    'prefill': torch.cuda.CUDAGraph(),
                    'decode': torch.cuda.CUDAGraph()}
        self.generate_cuda_graphs()

        # prepare some streams to use
        self.streams = [torch.cuda.Stream() for _ in range(18)]
        self.start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_trails)]
        self.end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_trails)]

    def generate_cuda_graphs(self):

        ## prepare cuda graphs for models
        recording_kwargs = {}
        self.extra_caches['enc_video'] = self.models['vision_enc'](self.caches['video'])
        with torch.cuda.graph(self.graphs['vision'], **recording_kwargs):
            self.extra_caches['enc_video'] = self.models['vision_enc'](self.caches['video'])
            _, self.extra_caches['enc_video'] = unpack(self.extra_caches['enc_video'], [torch.Size([8]), torch.Size([4])], 'b * d')
            self.extra_caches['enc_video'] = unpack(self.extra_caches['enc_video'], [torch.Size([1, 1])], '* n d')[0]
            self.extra_caches['enc_video'] = self.extra_caches['enc_video'][:, :6]
        self.graphs['vision'].replay()

        with torch.cuda.graph(self.graphs['audio'], **recording_kwargs):
            self.extra_caches['enc_audio'] = self.models['audio_enc'](self.caches['audio'])
            _, self.extra_caches['enc_audio'] = unpack(self.extra_caches['enc_audio'], [torch.Size([8]), torch.Size([4])], 'b * d')
            self.extra_caches['enc_audio'] = unpack(self.extra_caches['enc_audio'], [torch.Size([1, 1])], '* n d')[0]
            self.extra_caches['enc_audio'] = self.extra_caches['enc_audio'][:, :6]
        self.graphs['audio'].replay()

        with torch.cuda.graph(self.graphs['combiner'], **recording_kwargs):
            audio_and_video_tokens, _ = pack((self.extra_caches['enc_audio'], self.extra_caches['enc_video']), 'b n * d')
            audio_and_video_tokens, combine_ps = pack(audio_and_video_tokens, '* n d')
            self.extra_caches['combiner_out'] = self.models['combiner_enc'](audio_and_video_tokens)
        self.graphs['combiner'].replay()

        with torch.cuda.graph(self.graphs['encoder'], **recording_kwargs):
            combined_audio_video_tokens = self.extra_caches['combiner_out'][..., -3:, :]
            combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
            av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
            self.caches['av_embeddings'] = self.models['encoder'](av_encoder_input)
        self.graphs['encoder'].replay()

        ## Make cuda graph for the prefill phase
        kv_cache = None
        # print("self.caches['av_context'] shape: ", self.caches['av_context'].shape)
        # out, new_cache = self.models['decoder'].wrapped_decoder.make_graph(text_tokens, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
        with torch.cuda.graph(self.graphs['prefill'], **recording_kwargs):
            out, new_cache = self.models['decoder'].wrapped_decoder.make_graph(
                self.caches['text'], 
                seq_len = self.text_max_seq_len, 
                context = self.caches['av_context'], 
                kv_cache = kv_cache)
        self.graphs['prefill'].replay()
        torch.cuda.synchronize()

        ## Make cuda graph for the decode phase
        kv_cache = new_cache
        # out, new_cache = self.models['decoder'].wrapped_decoder.make_graph(single_token, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
        with torch.cuda.graph(self.graphs['decode'], **recording_kwargs):
            out, new_cache = self.models['decoder'].wrapped_decoder.make_graph(
                self.caches['single_token'], 
                seq_len = self.text_max_seq_len, 
                context = self.caches['av_context'], 
                kv_cache = kv_cache)
        self.graphs['decode'].replay()
        torch.cuda.synchronize()

    def generate_extra_cuda_graphs(self, ts_decode_num, ts_prefill_num):

        self.graphs['batch_decode'] = torch.cuda.CUDAGraph()
        # ts_decode_num = 1
        batch_single_token = torch.randint(0, 256, (ts_decode_num, 1)).to("cuda")
        self.caches['batch_single_token'] = batch_single_token
        batch_av_context = torch.randn(ts_decode_num, 18, 512).half().to("cuda")
        self.caches['batch_av_context'] = batch_av_context

        k_cache_trans = torch.randn(ts_decode_num, 8, 128, 64).half().to("cuda")
        v_cache_trans = torch.randn(ts_decode_num, 8, 128, 64).half().to("cuda")
        trans_cache = [k_cache_trans, v_cache_trans]

        recording_kwargs = {}
        with torch.cuda.graph(self.graphs['batch_decode'], **recording_kwargs):
            out, new_cache = self.models['decoder'].wrapped_decoder.make_graph(
                self.caches['batch_single_token'], 
                seq_len = self.text_max_seq_len, 
                context = self.caches['batch_av_context'], 
                kv_cache = None,
                supercache = trans_cache)

        self.graphs['batch_decode'].replay()
        torch.cuda.synchronize()

    def run_V_pipeline_cuda_graphs(self):
        # [FIXME]: This function not tested yet
        print('graph_V_exec.')
        graphs = [torch.cuda.CUDAGraph() for _ in range(4)]
        tensors = sequential_V_exec(models, caches, graphs)
        encoded_video, encoded_audio, combiner_output, av_embeddings = tensors

        audio_tokens, video_tokens, text_tokens = caches
        vision_enc, audio_enc, combiner_enc, encoder, decoder = models
        audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context = double_caches
        vision_graph, audio_graph, combiner_graph, encoder_graph = graphs
        print('graph_V_exec prepared')

        streams = [torch.cuda.Stream() for _ in range(4)]
        stream_high_priority = torch.cuda.Stream(priority=-1)
        stream_low_priority = torch.cuda.Stream(priority=0)
        start_time = time.time()
        all_sliced = True
        for i in range(1500):
            round_id.put(i)
            if all_sliced:

                with torch.cuda.stream(streams[3]):
                    encoder_graph.replay()
                    # encoder_buffers[i%2].copy_(av_embeddings)
                    av_context = torch.cat((av_context[:, :15, :], encoder_buffers[i%2]), dim=1)

                with torch.cuda.stream(streams[0]):
                    vision_graph.replay()
                    #TODO: We have to know the overhead of this copy operation
                    # video_buffers[i%2].copy_(encoded_video)

                with torch.cuda.stream(streams[1]):
                    audio_graph.replay()
                    # audio_buffers[i%2].copy_(encoded_audio)

                with torch.cuda.stream(streams[2]):
                    combiner_graph.replay()
                    # combiner_buffers[i%2].copy_(combiner_output)

            else:
                with torch.cuda.stream(stream_high_priority):
                    vision_graph.replay()
                    audio_graph.replay()
                    combiner_graph.replay()

                with torch.cuda.stream(stream_low_priority):
                    encoder_graph.replay()
                    av_context = torch.cat((av_context[:, :15, :], encoder_buffers[i%2]), dim=1)

            torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"graph_V_exec Elapsed Time: {elapsed_time} seconds")


    def run_V_cuda_graphs(self, num_trails=1, required_sync=True):
        for i in range(num_trails):
            self.graphs['vision'].replay()
            self.graphs['audio'].replay()
            self.graphs['combiner'].replay()
            self.graphs['encoder'].replay()
            if required_sync:
                torch.cuda.synchronize()


    def run_L_cuda_graphs(self, num_trails=1, out_seq_len=64, required_sync=True):
        for i in range(num_trails):
            self.graphs['prefill'].replay()
            for token in range(out_seq_len-1):
                self.graphs['decode'].replay()
                if required_sync:
                    torch.cuda.synchronize()


    def run_V_pipeline_basic(self):
        # [FIXME]: This function not tested yet

        streams = [torch.cuda.Stream() for _ in range(5)]
        for i in range(20):
            round_id.put(i)
            # with round_id.get_lock():
            #     current_id = round_id.value
            #     round_id.value = current_id + 1
            with torch.cuda.stream(streams[0]):
                encoded_video = vision_enc(video_tokens)
                _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
                encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]
                video_buffers[i%2] = encoded_video

            with torch.cuda.stream(streams[1]):
                encoded_audio = audio_enc(audio_tokens)
                _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
                encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
                audio_buffers[i%2] = encoded_audio

            with torch.cuda.stream(streams[2]):
                audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
                audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')
                combiner_output = combiner_enc(audio_and_video_tokens)
                combiner_buffers[i%2] = combiner_output

            with torch.cuda.stream(streams[3]):
                combined_audio_video_tokens = combiner_output[..., -3:, :]
                combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
                av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
                av_embeddings = encoder(av_encoder_input)
                encoder_buffers[i%2] = av_embeddings
                av_context = torch.cat((av_context[:, :15, :], encoder_buffers[i%2]), dim=1)

            torch.cuda.synchronize()

    def run_V_basic(self):

        encoded_video = self.models['vision_enc'](self.caches['video'])
        _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]

        encoded_audio = self.models['audio_enc'](self.caches['audio'])
        _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]

        # Synchronize the video encoder and the audio encoder
        torch.cuda.synchronize()

        audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
        audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')
        combiner_output = self.models['combiner_enc'](audio_and_video_tokens)
        # combiner_buffers[i%2] = combiner_output

        combined_audio_video_tokens = combiner_output[..., -3:, :]
        combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
        av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
        self.caches['av_embeddings'] = self.models['encoder'](av_encoder_input)
        self.caches['av_context'] = torch.cat((self.caches['av_context'], self.caches['av_embeddings']), dim=1)

        torch.cuda.synchronize()

    def run_L_basic(self, out_seq_len=64):

        self.caches['av_context'] = torch.cat((self.caches['av_context'][:, :15, :], self.caches['av_embeddings']), dim=1)
        out = self.models['decoder'].wrapped_decoder.generate(self.caches['text'], seq_len = out_seq_len, context = self.caches['av_context'])
        torch.cuda.synchronize()


    def run_VL_mp(self, use_cuda_graphs=True, num_trails=1):
        # [BUG]: This function is not working
        # round_id = mp.Queue()
        # mp.set_start_method('spawn')
        if use_cuda_graphs:
            V_process = Process(target=self.run_V_cuda_graphs, args=(self, num_trails))
            LA_process = Process(target=self.run_L_cuda_graphs, args=(self, num_trails))
        else:
            V_process = Process(target=self.run_V_basic, args=(self, num_trails))
            LA_process = Process(target=self.run_L_basic, args=(self, num_trails))

        processes = [V_process, LA_process]
        for process in processes:
            process.start()

        for process in processes:
            process.join()


    def run_VL_ms(self):
        with torch.cuda.stream(self.streams[0]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
        with torch.cuda.stream(self.streams[1]):
            self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)


    def run_single_request(self, durations, i, worker_num):
        stream_id = i%worker_num
        req_start = time.time()
        print("Launch - {}".format(self.idx))
        with torch.cuda.stream(self.streams[stream_id]):
            self.run_V_cuda_graphs(num_trails=1, required_sync=False)
            self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)
        self.streams[stream_id].synchronize()
        duration = time.time() - req_start
        print("Request duration: {:.3f} ms - {}".format(duration*1000, self.idx))
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


    def run_single_V(self, num_trails):
        trail_expand = 10
        all_start = time.time()
        for i in range(num_trails * trail_expand):
            start = time.time()
            with torch.cuda.stream(self.streams[0]):
                self.run_V_cuda_graphs(num_trails=1, required_sync=False)
            self.streams[0].synchronize()
            duration = time.time() - start
            sleep_time = args.req_interval - duration
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Enc time: {:.2f}".format(time.time() - all_start))


    def run_single_L(self, num_trails):
        start = time.time()
        for i in range(num_trails):
            with torch.cuda.stream(self.streams[1]):
                self.run_L_cuda_graphs(num_trails=1, out_seq_len=args.decode_len+args.prefill_len, required_sync=False)

        duration = time.time() - start
        print("LLM time: {:.2f}".format(duration))
        print("Query duration: {:.2f} ms".format(duration/num_trails*1000))


    def run_parallel_req(self, num_trails):
        start = []
        durations = []
        threads = []
        for i in range(num_trails):
            thread = threading.Thread(target=self.run_single_request, args=(durations, i, 2))
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


    def run_as_a_worker(self):

        req_num = 0
        start_time = time.time()
        while True:
            if not self.task_queue.empty():
                request_id = self.task_queue.get()
                print(f"Worker {self.idx} processing request: {request_id}")

                req_start_time = time.time()
                self.run_single_request_v2(self.streams[0], self.start_events[req_num], self.end_events[req_num], True)
                req_num = req_num + 1
                req_duration = time.time() - req_start_time
            else:
                time.sleep(0.01)

            if req_num == 200:
                duration = time.time() - start_time
                print("200 durations: ", duration)


    def run_all_in_mp(self):
        # [FIXME]: This function not tested yet
        audio_tokens, video_tokens, text_tokens = caches
        vision_enc, audio_enc, combiner_enc, encoder, decoder = models
        audio_buffers, video_buffers, combiner_buffers, encoder_buffers = double_caches

        buffer_ids = [mp.Value('i', 0) for _ in range(3)] #[av-c, c-e, e-d]
        events = {'af': mp.Event(), # audio finished
                'vf': mp.Event(), # video finished
                'cb': mp.Event(), # combiner begin
                'cf': mp.Event(), # combiner finished
                'eb': mp.Event(), # encoder begin
                'ef': mp.Event(), # encoder finished
                'db': mp.Event(), # decoder begin
                'df': mp.Event(), # decoder finished
                }

        _audio_process = Process(target=audio_process, args=(events, audio_tokens, audio_buffers, buffer_ids, audio_enc))
        _video_process = Process(target=video_process, args=(events, video_tokens, video_buffers, buffer_ids, vision_enc))
        _combiner_process = Process(target=combiner_process, args=(events, buffer_ids, audio_buffers, video_buffers, combiner_buffers, combiner_enc))
        _encoder_process = Process(target=encoder_process, args=(events, buffer_ids, combiner_buffers, encoder_buffers, encoder))

        processes = [_audio_process, _video_process, _combiner_process, _encoder_process]
        # processes = [_audio_process]

        start_time = time.time()
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        torch.cuda.synchronize()
        print(f"Time for 10 timesteps: {elapsed_time} seconds")


    def run_ts(self, task_plan):
        if args.profile_mode == 'base':
            stream_id = 0
            for stage in task_plan:
                if stage == 'e':
                    with torch.cuda.stream(self.streams[stream_id]):
                        self.graphs['vision'].replay()
                        self.graphs['audio'].replay()
                        self.graphs['combiner'].replay()
                        self.graphs['encoder'].replay()
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
                    self.graphs['vision'].replay()
                    self.graphs['audio'].replay()
                    self.graphs['combiner'].replay()
                    self.graphs['encoder'].replay()


    def run_benchmarks(self,
                       mode: str,
                       use_cuda_graphs: bool,
                       num_trails: int,
                       sche_plan
                       ):
        if mode == 'seq' and use_cuda_graphs:
            durations = []
            for i in range(num_trails):
                self.run_single_request(durations, 0, worker_num=1)
            torch.cuda.synchronize()
            print("Query duration: {:.2f} ms".format(np.mean(durations)*1000))

        elif mode == 'seq' and not use_cuda_graphs:
            for i in range(num_trails):
                self.run_V_basic()
                self.run_L_basic()

        elif mode == 'pipe':
            # self.run_VL_mp(use_cuda_graphs, num_trails)
            start = time.time()
            for i in range(num_trails):
                self.run_VL_ms()
                torch.cuda.synchronize()
            duration = time.time() - start
            print("Query duration: {:.2f} ms".format(duration/num_trails*2*1000))

        elif mode == 'decouple':
            thread1 = threading.Thread(target=self.run_single_V, args=(num_trails,))
            thread2 = threading.Thread(target=self.run_single_L, args=(num_trails,))

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

        # multi threads, multi streams, graphs contend and is not a correct impl
        elif mode == 'parallel':
            assert(0)
            durations, total_duration = self.run_parallel_req(num_trails=num_trails)
            print("Query latency: {:.2f} ms".format(np.mean(durations)*1000))
            print("IN throughput: {:.2f}".format(1/args.req_interval))
            print("OUT throughput: {:.2f}".format(num_trails/total_duration))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))

        # single thread, multi streams, graphs contend and is not a correct impl
        elif mode == 'parallel_v2':
            assert(0)
            durations, total_duration = self.run_parallel_req_v2(num_trails=num_trails)
            print("Query latency: {:.2f} ms".format(np.mean(durations)))
            print("Query duration: {:.2f}".format(total_duration*1000/num_trails))

        elif mode == 'profile':
            sche_duration = {}
            ts_encode_num = 0
            ts_decode_num = 0
            ts_prefill_num = 0

            for task_plan in sche_plan:
                # prepare batch graph
                for stage in task_plan:
                    if stage == 'e':
                        ts_encode_num = ts_encode_num + 1
                    elif stage == 'p':
                        ts_prefill_num = ts_prefill_num + 1
                    elif stage == 'd':
                        ts_decode_num = ts_decode_num + 1

                if ts_decode_num != 0:
                    self.generate_extra_cuda_graphs(ts_decode_num, ts_prefill_num)

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
            return sche_duration


def mirasol_run(sche_plan=None, mode='profile'):
    print("Start Mirasol inference...")

    # torch.cuda.set_device(1)
    mp.set_start_method('spawn')

    res = None
    profiler.start()

    if mode=='profile':
        e = Mirasol_engine(DIM=args.dim, idx=0)
        res = e.run_benchmarks(mode='profile',
                               use_cuda_graphs=True,
                               num_trails=args.num_trails,
                               sche_plan=sche_plan)

    elif mode=='parallel_sp':
        worker_num = args.worker_num
        e = [Mirasol_engine(DIM=args.dim, idx=i) for i in range(worker_num)]
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

    elif mode == 'parallel_mp':
        # [BUG] multi-process workers mode is not working.
        # [ERROR] RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad, since autograd does not support crossing process boundaries.  If you just want to transfer the data, call detach() on the tensor before serializing (e.g., putting it on the queue)
        worker_num = args.worker_num
        e = [Mirasol_engine(DIM=args.dim, idx=i) for i in range(worker_num)]
        processes = [mp.Process(target=e[i].run_as_a_worker) for i in range(worker_num)]

        for process in processes:
            process.start()

        for i in range(args.num_trails):
            worker_id = i%worker_num
            time.sleep(args.req_interval)
            e[worker_id].task_queue.put(i)

        for process in processes:
            process.join()

    else:
        e = Mirasol_engine(DIM=args.dim, idx=0)
        e.run_benchmarks(mode=mode,
                         use_cuda_graphs=True,
                         num_trails=args.num_trails,
                         sche_plan=None)

    torch.cuda.synchronize()
    profiler.stop()

    print("Mirasol inference finished")
    return res


if __name__ == "__main__":
    mirasol_run()
    exit()
