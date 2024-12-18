import torch

import os
import sys
sys.path.append("../../../repos/x-transformers")

from vit.model import vision_transformer
from llm.model import palme

class RT2_engine:

    def __init__(self):

        # prepare caches for tensors
        text = torch.randint(0, 20000, (1, 1024)).to("cuda")
        img = torch.randn(1, 3, 256, 256).to("cuda")
        context = torch.randn(1, 64, 512).to("cuda")
        single_token = torch.randint(0, 20000, (1, 1)).to("cuda")
        self.caches = { 'text': text,
                        'img': img,
                        'context': context,
                        'single_token': single_token}

        # prepare models
        vit = vision_transformer().to("cuda")
        llm = palme().to("cuda")
        self.models = {'vit': vit, 
                       'llm': llm}

        # prepare cuda graphs
        self.cuda_graphs = {'encode': torch.cuda.CUDAGraph(),
                            'prefill': torch.cuda.CUDAGraph(),
                            'decode': torch.cuda.CUDAGraph()}
        self.generate_cuda_graphs()

    def generate_cuda_graphs(self):
        recording_kwargs = {}
        text_max_seq_len = 256

        ## Make cuda graph for the prefill phase
        # [BUG]: I have to run the following command once to make the cuda graph generated properly
        # [FIXME]: The output caches of the graphs have not been designed yet
        # [FIXME]: The decode phase is static, which is just an approximate
        out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                            seq_len = text_max_seq_len, 
                                                            context = self.caches['context'], 
                                                            kv_cache = None)
        with torch.cuda.graph(self.cuda_graphs['prefill'], **recording_kwargs):
            out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['text'], 
                                                                seq_len = text_max_seq_len, 
                                                                context = self.caches['context'], 
                                                                kv_cache = None)
        self.cuda_graphs['prefill'].replay()
        torch.cuda.synchronize()

        ## Make cuda graph for the decode phase
        with torch.cuda.graph(self.cuda_graphs['decode'], **recording_kwargs):
            out, new_cache = self.models['llm'].wrapped_decoder.make_graph(self.caches['single_token'], 
                                                                seq_len = text_max_seq_len, 
                                                                context = self.caches['context'], 
                                                                kv_cache = new_cache)
        self.cuda_graphs['decode'].replay()
        torch.cuda.synchronize()

        ## Make cuda graph for the vision encoder
        with torch.cuda.graph(self.cuda_graphs['encode'], **recording_kwargs):
            out = self.models['vit'](self.caches['img'])
        self.cuda_graphs['encode'].replay()
        torch.cuda.synchronize()

    def run_cuda_graphs(self, num_trails):
        for i in range(num_trails):
            self.cuda_graphs['encode'].replay()
            self.cuda_graphs['prefill'].replay()
            self.cuda_graphs['decode'].replay()
        torch.cuda.synchronize()

    def run_basic(self, num_trails):
        pass

    def run_benchmarks(self,
                       mode: str,
                       use_cuda_graphs: bool,
                       num_trails: int):
        if mode == 'seq' and use_cuda_graphs:
            self.run_cuda_graphs(num_trails)
        elif mode == 'pipeline':
            pass


def sequential_L_exec(llm):

    text_tokens = torch.randint(0, 20000, (1, 1024)).to("cuda")
    context = torch.randn(1, 64, 512).to("cuda")
    llm.wrapped_decoder.generate(text_tokens, seq_len=256, context=context)



def graph_seq_exec(models, cuda_graph=True):
    if cuda_graph:
        run_cuda_graphs()
    else:
        run_basic()


if __name__ == "__main__":

    print("Start RT2 inference...")

    e = RT2_engine()
    e.run_benchmarks(mode='seq',
                     use_cuda_graphs=True,
                     num_trails=100)

    print("Finished.")

