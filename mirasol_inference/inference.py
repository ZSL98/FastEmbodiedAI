import torch
import torch.nn.functional as F
import time
import torch.multiprocessing as mp
from cuda import cuda, cudart
from torch.multiprocessing import Process, Value, Array

import os
import sys
sys.path.append("../../../repos/x-transformers")

from encoder0.vision_encoder import vision_encoder
from encoder0.audio_encoder import audio_encoder
from encoder1.combiner import combiner
from llm.encoder import mirasol_encoder
from llm.decoder import mirasol_decoder

from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Dict, Any
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange



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
    DIM=512

    def __init__(self):

        # prepare caches for tensors
        text_tokens = torch.randint(0, 256, (1, 128)).to("cuda")
        audio_tokens = torch.randn(1, 12, DIM).to("cuda")
        video_tokens = torch.randn(1, 12, DIM).to("cuda")
        self.caches = {'text': text_tokens, 'audio': audio_tokens, 'video': video_tokens}

        # prepare models
        vision_enc = vision_encoder(dim=DIM).to("cuda")
        audio_enc = audio_encoder(dim=DIM).to("cuda")
        combiner_enc = combiner(dim=DIM).to("cuda")
        encoder = mirasol_encoder(dim=DIM).to("cuda")
        decoder = mirasol_decoder(dim=DIM).to("cuda")
        self.models = {'vision_enc': vision_enc, audio_enc, combiner_enc, encoder, decoder}

        # prepare cuda graphs
        self.cuda_graphs = {'vision': torch.cuda.CUDAGraph(),
                            'prefill': torch.cuda.CUDAGraph(),
                            'decode': torch.cuda.CUDAGraph()}
        self.generate_cuda_graphs()

    def generate_cuda_graphs(self):

        ## prepare cuda graphs for models
        recording_kwargs = {}
        audio_tokens, video_tokens, text_tokens = caches
        with torch.cuda.graph(vision_graph, **recording_kwargs):
            encoded_video = vision_enc(video_tokens)
            _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
            encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]
            encoded_video = encoded_video[:, :6]
        vision_graph.replay()

        with torch.cuda.graph(audio_graph, **recording_kwargs):
            encoded_audio = audio_enc(audio_tokens)
            _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
            encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
            encoded_audio = encoded_audio[:, :6]
        audio_graph.replay()

        with torch.cuda.graph(combiner_graph, **recording_kwargs):
            audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
            audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')
            combiner_output = combiner_enc(audio_and_video_tokens)
        combiner_graph.replay()

        with torch.cuda.graph(encoder_graph, **recording_kwargs):
            combined_audio_video_tokens = combiner_output[..., -3:, :]
            combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
            av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
            av_embeddings = encoder(av_encoder_input)
        encoder_graph.replay()

        

def sequential_V_exec(models, caches, graphs):

    audio_tokens, video_tokens, text_tokens = caches
    vision_enc, audio_enc, combiner_enc, encoder, decoder = models
    vision_graph, audio_graph, combiner_graph, encoder_graph = graphs

    encoded_video = vision_enc(video_tokens)
    _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
    # encoded_video = unpack(encoded_video, [torch.Size([1, 6])], '* n d')[0]
    encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]
    encoded_video = encoded_video[:, :6]

    encoded_audio = audio_enc(audio_tokens)
    _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
    # encoded_audio = unpack(encoded_audio, [torch.Size([1, 32])], '* n d')[0]
    encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
    encoded_audio = encoded_audio[:, :6]

    print("encoded_video: ", encoded_video.shape)
    print("encoded_audio: ", encoded_audio.shape)

    audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
    audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')

    combiner_output = combiner_enc(audio_and_video_tokens)
    combined_audio_video_tokens = combiner_output[..., -3:, :]
    combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
    av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')

    print("av_encoder_input: ", av_encoder_input.shape)
    av_embeddings = encoder(av_encoder_input)
    print("av_embeddings: ", av_embeddings.shape)

    # text_max_seq_len = 256
    # out = decoder.wrapped_decoder.generate(text_tokens, seq_len = text_max_seq_len, context = av_embeddings)
    # print("out: ", out.shape)

    ## prepare cuda graphs for models
    recording_kwargs = {}
    audio_tokens, video_tokens, text_tokens = caches
    with torch.cuda.graph(vision_graph, **recording_kwargs):
        encoded_video = vision_enc(video_tokens)
        _, encoded_video = unpack(encoded_video, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_video = unpack(encoded_video, [torch.Size([1, 1])], '* n d')[0]
        encoded_video = encoded_video[:, :6]
    vision_graph.replay()

    with torch.cuda.graph(audio_graph, **recording_kwargs):
        encoded_audio = audio_enc(audio_tokens)
        _, encoded_audio = unpack(encoded_audio, [torch.Size([8]), torch.Size([4])], 'b * d')
        encoded_audio = unpack(encoded_audio, [torch.Size([1, 1])], '* n d')[0]
        encoded_audio = encoded_audio[:, :6]
    audio_graph.replay()

    with torch.cuda.graph(combiner_graph, **recording_kwargs):
        audio_and_video_tokens, _ = pack((encoded_audio, encoded_video), 'b n * d')
        audio_and_video_tokens, combine_ps = pack([audio_and_video_tokens], '* n d')
        combiner_output = combiner_enc(audio_and_video_tokens)
    combiner_graph.replay()

    with torch.cuda.graph(encoder_graph, **recording_kwargs):
        combined_audio_video_tokens = combiner_output[..., -3:, :]
        combined_audio_video_tokens = unpack(combined_audio_video_tokens, combine_ps, '* n d')[0]
        av_encoder_input = rearrange(combined_audio_video_tokens, 'b ... d -> b (...) d')
        av_embeddings = encoder(av_encoder_input)
    encoder_graph.replay()

    torch.cuda.synchronize()

    vision_graph.replay()
    audio_graph.replay()
    combiner_graph.replay()
    encoder_graph.replay()

    torch.cuda.synchronize()

    return encoded_video, encoded_audio, combiner_output, av_embeddings


def sequential_LA_exec(models, caches, double_caches, decoder_graph, decoder_decode_graph):

    _, _, text_tokens = caches
    _, _, _, _, decoder = models
    _, _, _, _, av_context = double_caches

    text_max_seq_len = 256
    recording_kwargs = {}

    ## Make cuda graph for the prefill phase
    kv_cache = None
    out, new_cache = decoder.wrapped_decoder.make_graph(text_tokens, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
    with torch.cuda.graph(decoder_graph, **recording_kwargs):
        out, new_cache = decoder.wrapped_decoder.make_graph(text_tokens, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
    decoder_graph.replay()
    torch.cuda.synchronize()

    ## Make cuda graph for the decode phase
    single_token = torch.randint(0, 256, (1, 1)).to("cuda")
    kv_cache = new_cache
    out, new_cache = decoder.wrapped_decoder.make_graph(single_token, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
    with torch.cuda.graph(decoder_decode_graph, **recording_kwargs):
        out, new_cache = decoder.wrapped_decoder.make_graph(single_token, seq_len = text_max_seq_len, context = av_context, kv_cache = kv_cache)
    decoder_decode_graph.replay()
    torch.cuda.synchronize()

    return out, new_cache

def stream_V_exec(round_id, models, caches, double_caches):
    audio_tokens, video_tokens, text_tokens = caches
    vision_enc, audio_enc, combiner_enc, encoder, decoder = models
    audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context = double_caches
    vision_graph, audio_graph, combiner_graph, encoder_graph, decoder_graph = graphs

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

def stream_LA_exec(round_id, models, caches, double_caches):
    print("stream_LA_exec.")
    audio_tokens, video_tokens, text_tokens = caches
    vision_enc, audio_enc, combiner_enc, encoder, decoder = models
    audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context = double_caches
    print('stream_LA_exec prepared')

    text_max_seq_len = 256
    # while True:
    #     i = round_id.get()
    #     if i == 5:
    #         # There should exist a lock to protect av_context
    #         av_context = torch.cat((av_context[:, :15, :], encoder_buffers[(i+1)%2]), dim=1)
    #         # context = encoder_buffers[(i+1)%2]
    #         out = decoder.wrapped_decoder.generate(text_tokens, seq_len = text_max_seq_len, context = av_context)
    #         break

    av_context = torch.cat((av_context[:, :15, :], encoder_buffers[0]), dim=1)
    out = decoder.wrapped_decoder.generate(text_tokens, seq_len = text_max_seq_len, context = av_context)

def stream_parallel_exec(models, caches, double_caches):

    # round_id = mp.Value("i", 0)
    round_id = mp.Queue()
    V_process = Process(target=stream_V_exec,  args=(round_id, models, caches, double_caches))
    LA_process = Process(target=stream_LA_exec, args=(round_id, models, caches, double_caches))

    processes = [V_process, LA_process]
    for process in processes:
        process.start()

    for process in processes:
        process.join()


def graph_seq_exec(models, cuda_graph=True):
    

def graph_V_exec(round_id, models, caches, double_caches):
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


def graph_LA_exec(round_id, models, caches, double_caches):
    print('graph_LA_exec')
    decoder_graph = torch.cuda.CUDAGraph()
    decoder_decode_graph = torch.cuda.CUDAGraph()
    tensor, new_cache = sequential_LA_exec(models, caches, double_caches, decoder_graph, decoder_decode_graph)
    LA_streams = [torch.cuda.Stream() for _ in range(2)]
    
    with torch.cuda.stream(LA_streams[0]):
        for i in range(200):
            decoder_graph.replay()
        torch.cuda.synchronize()

    with torch.cuda.stream(LA_streams[1]):
        for j in range(200):
            decoder_decode_graph.replay()
    
    torch.cuda.synchronize()
    print("Empty cache.")
    torch.cuda.empty_cache()

    # torch.cuda.synchronize()
    # with torch.cuda.stream(LA_streams[1]):
    #     for i in range(200):
    #         decoder_decode_graph.replay()

    # audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context = double_caches
    # text_max_seq_len = 256
    # av_context = torch.cat((av_context[:, :15, :], encoder_buffers[0]), dim=1)
    # decoder_graph.replay()


def graph_parallel_exec(models, caches, double_caches):

    # round_id = mp.Value("i", 0)
    round_id = mp.Queue()
    # manager = mp.Manager()
    audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context = double_caches
    
    # shared_tensor = manager.dict()
    # shared_tensor[''] = 
    # shared_tensor[''] = 

    # V_process = Process(target=graph_V_exec,  args=(round_id, models, caches, double_caches))
    LA_process = Process(target=graph_LA_exec, args=(round_id, models, caches, double_caches))

    # processes = [V_process, LA_process]
    processes = [LA_process]
    # V_process.start()
    # time.sleep(1)
    LA_process.start()
    for process in processes:
        process.join()


def graph_seq_exec(models):



def parallel_exec(models, caches, double_caches):

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


if __name__ == "__main__":

    print("Start...")
    mp.set_start_method('spawn')
    torch.cuda.set_device(0)
    # os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
    dimension = 512

    ## prepare tensors (input)
    print('prepare tensors (input)')
    audio_tokens = torch.randn(1, 12, dimension).to("cuda")
    video_tokens = torch.randn(1, 12, dimension).to("cuda")
    text_tokens = torch.randint(0, 256, (1, 128)).to("cuda")
    caches = [audio_tokens, video_tokens, text_tokens]

    ## prepare intermediate double buffers (output)
    print('prepare intermediate double buffers (output)')
    audio_buffers = [torch.randn(1, 1, 4, dimension).to("cuda") for _ in range(2)]
    video_buffers = [torch.randn(1, 1, 4, dimension).to("cuda") for _ in range(2)]
    combiner_buffers = [torch.randn(1, 8, dimension).to("cuda") for _ in range(2)]
    encoder_buffers = [torch.randn(1, 3, dimension).to("cuda") for _ in range(2)]
    av_context = torch.randn(1, 18, dimension).to("cuda")
    double_caches = [audio_buffers, video_buffers, combiner_buffers, encoder_buffers, av_context]

    ## prepare intermediate buffer (input) for CUDAgraph
    print('prepare intermediate buffer (input) for CUDAgraph')
    encoded_video = torch.randn(1, 1, 4, dimension).to("cuda")
    encoded_audio = torch.randn(1, 1, 4, dimension).to("cuda")
    combined_audio_video_tokens = torch.randn(1, 8, dimension).to("cuda")
    av_embeddings = torch.randn(1, 3, dimension).to("cuda")

    ## prepare components
    print('prepare components')
    vision_enc = vision_encoder(dim=dimension).to("cuda")
    audio_enc = audio_encoder(dim=dimension).to("cuda")
    combiner_enc = combiner(dim=dimension).to("cuda")
    encoder = mirasol_encoder(dim=dimension).to("cuda")
    decoder = mirasol_decoder(dim=dimension).to("cuda")
    models = [vision_enc, audio_enc, combiner_enc, encoder, decoder]
    print('preparation finished')

    # sequential_V_exec(models, caches)
    # parallel_exec(models, caches, double_caches)
    # stream_parallel_exec(models, caches, double_caches)
    graph_parallel_exec(models, caches, double_caches)

    print('Finished.')