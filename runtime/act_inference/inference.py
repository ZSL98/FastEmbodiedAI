import torch
import numpy as np

import argparse
from encoder.model import act_encoder
from llm.model import act_decoder

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', action='store', type=str, help='task_name')
    parser.add_argument('--seed', action='store', type=int, help='seed')
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs')
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser

class ACT_engine:

    def __init__(self, args):

        # prepare caches for tensors
        camera_imgs = torch.randn(1, 4, 3, 480, 640).to("cuda")
        qpos = torch.randn(1, 14).to("cuda")
        src = torch.randn(1, 256, 15, 80).to("cuda")
        pos = torch.randn(1, 256, 15, 80).to("cuda")
        latent_input = torch.randn(1, 256).to("cuda")
        proprio_input = torch.randn(1, 256).to("cuda")
        self.caches = {'qpos': qpos,
                       'camera_imgs': camera_imgs,
                       'src': src,
                       'pos': pos,
                       'latent_input': latent_input,
                       'proprio_input': proprio_input}

        # prepare models
        vision_encoder = act_encoder(args).to("cuda")
        llm = act_decoder(args).to("cuda")
        self.models = {'vision_encoder': vision_encoder, 
                       'llm': llm}

        # prepare cuda graphs
        self.cuda_graphs = {'encode': torch.cuda.CUDAGraph(),
                            'prefill': torch.cuda.CUDAGraph(),
                            'decode': torch.cuda.CUDAGraph()}

    def generate_cuda_graphs(self):
        recording_kwargs = {}
        # [TODO]: The graph is not splitted properly
        with torch.cuda.graph(self.cuda_graphs['prefill'], **recording_kwargs):
            pass

    def run_cuda_graphs(self):
        pass

    def run_basic(self):
        src, pos, latent_input, proprio_input = self.models['vision_encoder'](self.caches['qpos'], 
                                                                              self.caches['camera_imgs'])

        # [TODO]: This copy operation may not be efficient
        self.caches['src'].copy_(src)
        self.caches['pos'].copy_(pos)
        self.caches['latent_input'].copy_(latent_input)
        self.caches['proprio_input'].copy_(proprio_input)
        out = self.models['llm'](self.caches['src'], 
                                 self.caches['pos'], 
                                 self.caches['latent_input'], 
                                 self.caches['proprio_input'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    print("Start ACT inference...")

    e = ACT_engine(args)
    e.run_basic()

    print("Finished.")
