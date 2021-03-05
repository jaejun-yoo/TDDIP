import os
import json
import h5py
import shutil
import importlib
import torch
import numpy as np
import time
import torch.autograd.profiler as profiler 

#profiling tools
from thop import profile 

import scipy.io as sio
from option import get_option
from solver import Solver
from utils.common_utils import get_input_manifold

from IPython.core.debugger import set_trace

def main():

    opt = get_option()

    if torch.cuda.is_available(): 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
        torch.cuda.set_device(1)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        print("Current device: idx%s | %s" %(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
        
    torch.manual_seed(opt.seed)

    module = importlib.import_module("models.{}".format(opt.model.lower()))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load dataset 
    seq=np.squeeze(sio.loadmat(opt.fname)['data'])
    Nfr = np.shape(seq)[2]*opt.num_cycle
    img_size=np.shape(seq)[0]

    #make a batch of inputs
    net_input_set = get_input_manifold(opt.input_type, opt.num_cycle, Nfr).to(dev)

    if opt.input_type.endswith('mapping'):
        mapnet = module.MappingNet(opt).to(dev)            
        net_input_set = mapnet(net_input_set).reshape((Nfr,1,
                                                img_size//opt.up_factor, 
                                                img_size//opt.up_factor))

    idx_fr=np.random.randint(0, Nfr)
    idx_frs = range(min(idx_fr, Nfr-opt.batch_size), min(idx_fr+opt.batch_size, Nfr))
    net_input_z = torch.autograd.Variable(net_input_set[idx_frs,:,:,:],requires_grad=True)

    net = module.Net(opt).to(dev)

    #measure flops in one forward pass
    macs, params = profile(net, inputs=(net_input_z, ))

    #measure time and memory usage during both forward and backward passes
    with profiler.profile(profile_memory=True, use_cuda=torch.cuda.is_available()) as prof:
        with profiler.record_function("model_inference"):
            out = torch.linalg.norm(net(net_input_z))
            out.backward()

    print()
    print("- - - TIME AND MEMORY - - - ")
    print(prof.key_averages().table(sort_by="cpu_time_total",row_limit=10))
    print(" - - - TOTAL FLOPS (FORWARD PASS) - - - ")
    print("GMACs: ", macs*10e-9)
    print("GFLOPs: ", 2*macs*10e-9)

if __name__ == "__main__":
    main()
