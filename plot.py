import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.common_utils import get_input_manifold
import argparse
import os
import scipy.io as sio
import json
import importlib
import torch

"""
Script for plotting results. Define the logging directory containing model statedicts with --logdir, 
and define the particular checkpoint with --checkpoint.

Select the --plot_type as 'residual' for an image of the absolute residual with a ground truth frame --frame_index, 
and 'interpolate' for an image of an interpolated image between --interp_index and --interp_index + 1.
"""

def interp_input_line(set_, index, alpha = 0.5):

    """
    straight line interpolation between set_[index] and set_[index + 1]
    used for subframe sampling
    """

    point_a = set_[index]
    point_b = set_[index + 1] 

    interp = alpha*point_a + (1.0 - alpha)*point_b

    return interp

def collect_ground_truth(opt):   
    fname = opt.fname
    num_cycle = opt.num_cycle
    Nfibo = opt.Nfibo
    seq=np.squeeze(sio.loadmat(fname)['data']) # numpy array (128, 128, 23, 32), complex128, kt-space data
    coil=sio.loadmat(fname)['b1'].astype(np.complex64) #  numpy array, coil sensitivity
    coil = np.transpose(coil,(2,0,1)) # (32, 128, 128)

    Nc=np.shape(seq)[-1] # 32 number of coils
    Nvec=np.shape(seq)[0]*2 # 256 radial sampling number (virtual k-space)
    Nfr = np.shape(seq)[2]*num_cycle # 23 number of frames * num_cycle (13)
    img_size=np.shape(seq)[0] # 128        

    gt_cartesian_kt = seq[...,np.newaxis].astype(np.complex64) # (128, 128, 23, 32, 1), complex64, kt-space data 
    gt_cartesian_kt_ri = np.concatenate((np.real(gt_cartesian_kt),np.imag(gt_cartesian_kt)),axis=-1) # (128, 128, 23, 32, 2), float32, kt-space data 
    gt_cartesian_kt_ri = np.transpose(gt_cartesian_kt_ri,(3,2,0,1,4)) # (32, 23, 128, 128, 2), kt-space data
    gt_cartesian_kt_ri = np.concatenate([gt_cartesian_kt_ri]*num_cycle,axis = 1) # (32, 23*num_cycle, 128, 128, 2), kt-space data
    gt_cartesian_kt = np.concatenate([gt_cartesian_kt]*num_cycle,axis=2) # (128, 128, 23*num_cycle, 32, 1), complex64, kt-space data 


    w1=np.linspace(1,0,Nvec//2) # [1,...,0] length 128
    w2=np.linspace(0,1,Nvec//2) # [0,...,1] length 128
    w=np.concatenate((w1,w2),axis=0)[np.newaxis,np.newaxis] # [1,...0,0,...,1] (1, 1, 256)
    wr=np.tile(w,(Nc,Nfibo,1)) # (32, 13, 256) repeated w
    denc = wr.astype(np.complex64)
    
    # For visualization: GT full sampled images
    gt_cartesian = gt_cartesian_kt.transpose(3,2,0,1,4) # (32, 23*num_cycle, 128, 128, 1)
    gt_cartesian = gt_cartesian[:,:,:,:,0] # (32, 23*num_cycle, 128, 128)
    gt_cartesian_img = np.zeros((img_size,img_size,Nfr)) # (128, 128, 23*num_cycle)
    for idx_fr in range(Nfr):
        curr_gt_cartesian_img = np.sqrt((abs(gt_cartesian[:,idx_fr,:,:]*coil)**2).mean(0))
        curr_gt_cartesian_img -= curr_gt_cartesian_img.min()
        curr_gt_cartesian_img /= curr_gt_cartesian_img.max()
        gt_cartesian_img[:,:,idx_fr] = curr_gt_cartesian_img

    return gt_cartesian_img

if torch.cuda.is_available(): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
    torch.cuda.set_device(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    print("Current device: idx%s | %s" %(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default=".")
parser.add_argument("--checkpoint", type=str, default="1000.pt")
parser.add_argument("--plot_type", type=str, default="residual")
parser.add_argument("--frame_index", type=int, default=100)
parser.add_argument("--out_fname", type=str, default="img.jpg")
opt = parser.parse_args()

checkpoint = torch.load(os.path.join(opt.logdir, opt.checkpoint), map_location=lambda storage, loc: storage)

#load parameters from this directory
with open(os.path.join(opt.logdir, 'myparam.json'), 'r') as f:
    data = json.load(f)
    v = vars(opt)
    for i in data.keys():
        #add keyvals to opt
        v[i] = data[i]
    
#create a network with the same properties 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
module = importlib.import_module("models.{}".format(opt.model.lower())) 

net = module.Net(opt).to(dev)
mapnet = module.MappingNet(opt).to(dev) 

net.load_state_dict(checkpoint['net_state_dict']) 
mapnet.load_state_dict(checkpoint['mapnet_state_dict'])
input_set = checkpoint['net_input_set'].to(dev) #mapped inputs from before
helix_inputs = get_input_manifold(opt.input_type, opt.num_cycle, 299).to(dev) #inputs to mapping network
ground_truth = collect_ground_truth(opt)

# TODO: These are currently hardcoded values. They 
# Nframes = 299
# Img size = 128

if (opt.plot_type == 'interpolate'):

    frame_index = opt.frame_index #select frame 
    interp_input = interp_input_line(helix_inputs, frame_index) #generate interpolated input 
    map_output = mapnet(interp_input) #pass through mapping net 

    out = net(map_output.reshape((1, 1, 128//opt.up_factor, 128//opt.up_factor))).detach().cpu().numpy()[0]
    frm = np.sqrt(out[0,:,:]**2+out[1,:,:]**2)

    ax = plt.subplot()
    ax.imshow(frm, cmap='gray')
    plt.show()
    plt.savefig(opt.out_fname, dpi=200)

elif(opt.plot_type == 'residual'):

    frame_index = opt.frame_index
    net_input_z = torch.autograd.Variable(input_set[frame_index,:,:,:],requires_grad=False) #this is the mapped input : not from helix
    out = net(net_input_z.reshape((1, 1, 128//opt.up_factor, 128//opt.up_factor))).detach().cpu().numpy()[0]

    ref = ground_truth[:,:,frame_index]
    out=np.sqrt(out[0,:,:]**2 + out[1,:,:]**2)
    out -= out.min()
    out /= out.max() 
    diff = np.absolute(out - ref)

    ax = plt.subplot()
    im = ax.imshow(diff, vmin=0.0, vmax=1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.show()
    plt.savefig(opt.out_fname,  dpi=200)
