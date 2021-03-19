import argparse
import os
import numpy as np
from datetime import datetime

def parse_args(): 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # models
    parser.add_argument("--model", type=str, default="basicnet")
    parser.add_argument("--opt_over", type=str, default="net")
    parser.add_argument("--input_type", type=str, default="helix+mapping")

    # dataset
    parser.add_argument("--dataset", type=str, default="real") # retro | real
    parser.add_argument("--fname", type=str, default="dataset1.mat") # series11_2.mat p =~18
    parser.add_argument("--Nfibo", type=int, default=5) # 13: retro | 5: real
    parser.add_argument("--num_cycle", type=int, default=13) 
    
    # training setups
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=2000) # scheduler
    parser.add_argument("--gamma", type=float, default=0.5) # scheduler
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)    

    # misc
    parser.add_argument("--PLOT", action="store_true")    
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
        opt.ndf = 128
        opt.up_factor = 32
        opt.num_ups = int(np.log2(opt.up_factor))
        opt.need_tanh = False
        opt.need_bias = False
        opt.need_convT = False
        opt.upsample_mode = 'nearest'        
        opt.latent_dim = 3
        opt.style_dim = 64
        opt.depth = 0
        opt.description = '%s: basic net, %s, %s cycles, Nfibo=%s' % (opt.dataset, opt.input_type, opt.num_cycle, opt.Nfibo)
    else:
        raise NotImplementedError('what is it?')
        
    if opt.isresume is not None:
        opt.ckpt_root = os.path.dirname(opt.isresume)

def get_option():              
    opt = parse_args()    
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d_%H%M%S")      
    opt.ckpt_folder = opt.dataset+"_"#+ curr_time    
    opt.ckpt_root = "./logs/"+opt.ckpt_folder      
    make_template(opt)
    
    return opt
