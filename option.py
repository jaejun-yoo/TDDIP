import argparse
import os
import numpy as np
from datetime import datetime

def parse_args():    
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d_%H%M%S")
    ckpt_folder = "retro_"+ curr_time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # models
    parser.add_argument("--model", type=str, default="basicnet")
    parser.add_argument("--opt_over", type=str, default="net")
    parser.add_argument("--input_type", type=str, default="helix+mapping")

    # dataset
    parser.add_argument("--dataset", type=str, default="Retrospective")
    parser.add_argument("--fname", type=str, default="cardiac32ch_b1.mat")
    parser.add_argument("--Nfibo", type=int, default=13) # 13: retro | 5: real
    parser.add_argument("--num_cycle", type=int, default=13) 
    
    # training setups
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=2000) # scheduler
    parser.add_argument("--gamma", type=float, default=0.5) # scheduler
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)    

    # misc
    parser.add_argument("--PLOT", action="store_true")
    parser.add_argument("--ckpt_root", type=str, default="./logs/"+ckpt_folder)
    parser.add_argument("--isresume", type=str , default=None) # ckpt_filepath, e.g., "./logs/retro_20210221_142941/500.pt"
    parser.add_argument("--save_period", type=int, default=500)
    parser.add_argument("--description", type=str, default="")

    return parser.parse_args()


def make_template(opt):

    # model
    if "basicnet" in opt.model:
        opt.input_depth = 1
        opt.num_output_ch = 2
        opt.Nr = 1
        opt.ndf = 128 #original 128
        opt.up_factor = 16
        opt.num_ups = int(np.log2(opt.up_factor))
        opt.need_tanh = False
        opt.need_bias = False
        opt.need_convT = False
        opt.upsample_mode = 'nearest'        
        opt.latent_dim = 3
        opt.style_dim = 64
        opt.depth = 1
        opt.description = '%s: basic net, %s, %s cycles, Nfibo=%s' % (opt.dataset, opt.input_type, opt.num_cycle, opt.Nfibo)
    elif "deep_decoder" in opt.model:
        opt.latent_dim = 3
        opt.style_dim = 64
        opt.depth = 2
        opt.learn_bn = True
        opt.inp_ch = 1 #must 
        opt.num_output_ch = 2 #number of output channels for the image
        opt.need_sigmoid = False
        opt.up_factor = 16
        opt.num_ups = int(np.log2(opt.up_factor))
        opt.channel_nums = [128]*(opt.num_ups + 1)
    elif "baseline_decoder_hybrid" in opt.model:
        opt.input_depth = 1
        opt.num_output_ch = 2
        opt.Nr = 1 #number of intermediate layers
        opt.up_factor = 16
        opt.num_ups = int(np.log2(opt.up_factor))
        opt.latent_dim = 3
        opt.style_dim = 64
        opt.depth = 1
        opt.num_channels = [1,32,16,16,16,128,opt.num_output_ch]
        opt.layer_mask=[False]*opt.num_ups
    else:
        raise NotImplementedError('what is it?')
        
    if opt.isresume is not None:
        opt.ckpt_root = os.path.dirname(opt.isresume)

def get_option():    
    opt = parse_args()
    make_template(opt)
    return opt
