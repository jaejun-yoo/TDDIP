import argparse
import os
import sys
import json
import h5py
import shutil
import numpy as np
from datetime import datetime

def parse_args():        
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=0)

    # models
    parser.add_argument("--model", type=str, default="basicnet")
    parser.add_argument("--opt_over", type=str, default="net")
    parser.add_argument("--input_type", type=str, default="helix+mapping")
    parser.add_argument("--latent_dim", type=int, default=3) # manifold
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network') # mapping net
    parser.add_argument("--style_size", type=int, default=8) # mapping net
    parser.add_argument("--depth", type=int, default=0) # mapping net
    parser.add_argument("--Nr", type=int, default=1)
    parser.add_argument("--ndf", type=int, default=128)
    parser.add_argument("--input_nch", type=int, default=1)
    parser.add_argument("--output_nch", type=int, default=2)
    parser.add_argument("--need_bias", action="store_true")
    parser.add_argument("--up_factor", type=int, default=16)
    parser.add_argument("--upsample_mode", type=str, default="nearest")    

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
    parser.add_argument("--noPLOT", action="store_true")
    parser.add_argument("--isdev", action="store_true")  
    parser.add_argument("--istest", action="store_true") 
    parser.add_argument("--isresume", type=str , default=None) # ckpt_filepath, e.g., "./logs/retro_20210221_142941/500.pt"
    parser.add_argument("--ckpt_root", type=str, default="./logs/")
    parser.add_argument("--save_period", type=int, default=100)
    parser.add_argument("--memo", type=str, default="")

    return parser.parse_args()


def make_template(opt):
    if opt.isdev:
        ckpt_folder = "retro_"
    else:
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d_%H%M%S")    
        ckpt_folder = "retro_{}".format(curr_time)

    opt.ckpt_root = opt.ckpt_root + ckpt_folder
    print(opt.ckpt_root)
    os.makedirs(opt.ckpt_root, exist_ok=True)

    with open(os.path.join(opt.ckpt_root, 'myparam.json'), 'w') as f:
        json.dump(vars(opt), f)

    with open(opt.ckpt_root+"/command_line_log.txt", "w") as log_file:
        log_file.write("python %s" % " ".join(sys.argv))

    shutil.copy(os.path.join(os.getcwd(),__file__),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'solver.py'),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'option.py'),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'models','{}.py'.format(opt.model)),opt.ckpt_root) 
        
    # model
    if "basicnet" in opt.model:
        opt.memo = '%s: basic net, %s, %s cycles, Nfibo=%s' % (opt.dataset, opt.input_type, opt.num_cycle, opt.Nfibo) + opt.memo    
    else:
        raise NotImplementedError('what is it?')


def get_option():    
    opt = parse_args()
    if opt.isresume is None:
        make_template(opt)
    else:
        print('Resumed from ' + opt.isresume)   
        with open(os.path.join(os.path.dirname(opt.isresume), 'myparam.json'), 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            t_args.isresume = opt.isresume
            t_args.ckpt_root = os.path.dirname(opt.isresume)
            t_args.istest = opt.istest
            opt = t_args    
            
    return opt
