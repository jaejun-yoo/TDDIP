import os
import json
import h5py
import shutil
import importlib
import torch
from option import get_option
from solver import Solver

from IPython.core.debugger import set_trace
def main():
    opt = get_option()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
    torch.cuda.set_device(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    torch.manual_seed(opt.seed)

    print("Current device: idx%s | %s" %(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))

    module = importlib.import_module("models.{}".format(opt.model.lower()))

    if opt.isresume is None:
        print(opt.ckpt_root)
        os.makedirs(opt.ckpt_root, exist_ok=True)
        with open(os.path.join(opt.ckpt_root, 'myparam.json'), 'w') as f:
            json.dump(vars(opt), f)
            
        shutil.copy(os.path.join(os.getcwd(),__file__),opt.ckpt_root)
        shutil.copy(os.path.join(os.getcwd(),'solver.py'),opt.ckpt_root)
        shutil.copy(os.path.join(os.getcwd(),'option.py'),opt.ckpt_root)
    else:
        print('Resumed from ' + opt.isresume)   
        
    solver = Solver(module, opt)
    solver.fit()

if __name__ == "__main__":
    main()
