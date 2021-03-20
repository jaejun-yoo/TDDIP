import os
import json
import time
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.common_utils import *
from Mypnufft_mc_func_grasp_norm_v2 import *

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.core.debugger import set_trace


class Solver():
    def __init__(self, module, opt):
        
        self.opt = opt
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
        
        self.prepare_dataset()
        self.writer = SummaryWriter(opt.ckpt_root)
        
        self.net = module.Net(opt).to(self.dev)
        self.net_input_set = get_input_manifold(opt.input_type, opt.num_cycle, self.Nfr).to(self.dev)
        
        p = get_params(opt.opt_over, self.net, self.net_input_set)
        
        if opt.input_type.endswith('mapping'):
            self.mapnet = module.MappingNet(opt).to(self.dev)            
            print('... adding params of mapping network ...')
            p += self.mapnet.parameters()
            self.net_input_set = self.mapnet(self.net_input_set).reshape((self.Nfr,1,
                                                    self.img_size//self.opt.up_factor, 
                                                    self.img_size//self.opt.up_factor))
            
        # Compute number of parameters          
        s  = sum([np.prod(list(pnb.size())) for pnb in p]);
        print ('# params: %d' % s)

        self.step = 0
        self.optimizer = torch.optim.Adam(p, lr=opt.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=opt.step_size, 
                                                    gamma=opt.gamma)
        if opt.isresume is not None:
            self.load(opt.isresume)
        
        self.loss_fn = nn.MSELoss()
        

        self.t1, self.t2 = None, None
        self.best_psnr, self.best_psnr_step = 0, 0
        self.best_ssim, self.best_ssim_step = 0, 0

    def fit(self):
        opt = self.opt
        batch_size = opt.batch_size   
        Nc = self.Nc
        Nfr = self.Nfr
        Nvec = self.Nvec
        Nfibo = opt.Nfibo
        coil = self.coil
        denc = self.denc
        net_input_set = self.net_input_set
        real_radial_ri_ts = self.real_radial_ri_ts    
        step = self.step
        
        self.t1 = time.time()
        while step < opt.max_steps:
            # randomly pick frames to train (batch, default = 1)
            idx_fr=np.random.randint(0, Nfr)
            idx_frs = range(min(idx_fr, Nfr-batch_size), min(idx_fr+batch_size, Nfr))

            net_input_z = torch.autograd.Variable(net_input_set[idx_frs,:,:,:],requires_grad=False) # net_input_set: e.g., torch.Size([1400, 1, 8, 8]) 

            out_sp = self.net(net_input_z) # e.g., spatial domain output (img) torch.Size([batch_size, 2, 256, 256)
            out_sp = out_sp.permute(0,2,3,1)
            out_kt = []
            gt_kt = []

            for idx_b in range(batch_size):
                idx_fr = idx_frs[idx_b]
                angle = self.set_ang[np.maximum(0,idx_fr-opt.fib_st):np.minimum(Nfr,idx_fr+opt.fib_ed),:,:] # (5, 512, 2)             
                gt_kt.append(real_radial_ri_ts[0,:,np.maximum(0,idx_fr-opt.fib_st):np.minimum(Nfr-1,idx_fr+opt.fib_ed),:,:].reshape(-1,2)) # real_radial_ri_ts: torch.Size([1, 20, 1400, 512, 2]) => torch.Size([51200, 2]), 51200 = 20x5x512
                self.mynufft.X=out_sp[idx_b,:,:,:]
                out_kt.append(self.mynufft(angle.reshape((-1,2)),angle.shape[0],Nvec,Nc,coil,denc).reshape(-1,2))

            out_kt = torch.cat(out_kt)
            gt_kt = torch.cat(gt_kt)

            total_loss = self.loss_fn(gt_kt[...,0],out_kt[...,0])+ self.loss_fn(gt_kt[...,1],out_kt[...,1])

            total_loss *= (self.img_size)**2     
            self.optimizer.zero_grad()
            total_loss.backward()   
            self.optimizer.step()
            self.scheduler.step()
            
            self.writer.add_scalar('total loss/training_loss (mse)', total_loss, step)
            
            if opt.PLOT and (step % opt.save_period == 0 or step == opt.max_steps-1):
                self.summary_and_save(step,out_sp, idx_fr)

            step += 1
            self.step = step            
        
        self.writer.close()   
        self.save_video()
                
    def summary_and_save(self, step, out_sp, idx_fr):        
        max_steps = self.opt.max_steps
        
        # For visualization
        idx_fr = np.random.randint(0, self.Nfr)
        out_abs=(out_sp[0,:,:,0]**2+out_sp[0,:,:,1]**2)**.5 
        out_abs = out_abs-out_abs.min()
        out_abs = out_abs/out_abs.max()
        
        real_radial_img_ts = self.real_radial_img.to(self.dev).float()

        images_grid = torch.cat([torch.flip(real_radial_img_ts[None],[1]),torch.flip(out_abs[None],[1])],dim=2)
        images_grid = F.interpolate(images_grid.unsqueeze(0), scale_factor = 4).squeeze(0)
        self.writer.add_image('recon_image', images_grid, step)
        self.t2 = time.time()
        self.save(step)

        curr_lr = self.scheduler.get_lr()[0]
        eta = (self.t2-self.t1) * (max_steps-step) /self.opt.save_period / 3600
        print("[{}/{}] LR: {}, ETA: {:.1f} hours"
            .format(step, max_steps, curr_lr, eta))

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):        
        raise NotImplementedError("Evaluate function is not implemented yet.")
        return 

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.net_input_set = checkpoint['net_input_set'].to(self.dev)
        self.net.load_state_dict(checkpoint['net_state_dict']) 
        self.net.to(self.dev)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.step_size, gamma=self.opt.gamma) 
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']+1
        
        
    def save(self, step):        
        print('saving ... ')
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        ckptdict = {
                'step': step,
                'net_input_set': self.net_input_set,
                'net_state_dict': self.net.state_dict(), 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                }
        best_scores = {
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_psnr_step': self.best_psnr_step,
                'best_ssim_step': self.best_ssim_step,
                }
        if self.opt.input_type.endswith('mapping'):            
            ckptdict['mapnet_state_dict']: self.mapnet.state_dict() 
                
        torch.save(ckptdict, save_path)
        with open(os.path.join(self.opt.ckpt_root, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)
        
    def save_video(self):
        
        #h5 save
        ims = []
        inp = []

        for idx_fr in range(self.Nfr):
            net_input_fr=self.net_input_set[idx_fr,:,:,:][np.newaxis,:,:,:]
            out_HR_np = torch_to_np(self.net(net_input_fr))
            net_input_fr=torch_to_np(net_input_fr)
            ims.append(out_HR_np)
            inp.append(net_input_fr)

        f = h5py.File(os.path.join(self.opt.ckpt_root, 'final.h5'),'w')
        f.create_dataset('input',data=inp,dtype=np.float32)
        f.create_dataset('data',data=ims,dtype=np.float32)
        f.create_dataset('angle',data=self.set_ang,dtype=np.float32)
        f.close()
        print('h5 file saved.')
        
        print('creating video.')
        fig = plt.figure(figsize=(10, 10))
        vid = []
        for idx_fr in range(self.Nfr):
            tmp_ims=np.sqrt(ims[idx_fr][0,:,:]**2+ims[idx_fr][1,:,:]**2)
            tmp_ims -= tmp_ims.min()
            tmp_ims /= tmp_ims.max()               
            ttl = plt.text(128, -5, idx_fr, horizontalalignment='center', fontsize = 20)
            vid.append([plt.imshow(np.flip(tmp_ims,0), animated=True, cmap = 'gray', vmax=0.5),ttl])
        ani = animation.ArtistAnimation(fig, vid, interval=50, blit=True, repeat_delay=1000)

        ani.save(self.opt.ckpt_root+'/final_video.mp4')
        print('video saved')
    
    def prepare_dataset(self):  
        fname = self.opt.fname
        num_cycle = self.opt.num_cycle
        Nfibo = self.opt.Nfibo
        # ========================= #
        # # == For spoke-sharing == #
        # ========================= #        
        self.opt.fib_st=Nfibo//2
        if np.mod(Nfibo,2)==0:
            self.opt.fib_ed=Nfibo//2
        else:
            self.opt.fib_ed=Nfibo//2+1 
        
        seq=1e3*np.squeeze(sio.loadmat(fname)['rawData']).astype(np.complex64) # numpy array (512, 1600, 20), complex128, kt-space data
        seq = seq[:,200:,:] # Removing the first 200 data frame, which have bleaching noise (by Kyong)
        coil=sio.loadmat(fname)['C'].astype(np.complex64) #  numpy array, coil sensitivity
        coil = np.transpose(coil,(2,0,1)) # (20, 256, 256)

        Nc=np.shape(seq)[-1] # 20 number of coils
        Nvec=np.shape(seq)[0] # 512 radial sampling number (virtual k-space)
        Nfr = np.shape(seq)[1] # 1400 number of frames
        img_size=np.shape(coil)[-1] # 256        
        
        denc=sio.loadmat(fname)['W'] # (512, 1600)
        denc=denc[:,0] # all are same, so just take one.
        real_radial = np.transpose(seq,(2,1,0))[...,np.newaxis]
        real_radial_ri = np.concatenate((np.real(real_radial),np.imag(real_radial)),axis=3) # real and imaginary
    
        real_radial_ri*=np.tile(denc.astype(np.float32).reshape(1,1,-1,1),[Nc,Nfr,1,2])
        real_radial_ri_ts=np_to_torch(real_radial_ri).to(self.dev).detach()

        # 111.246 degree - golden angle | 23.63 degree - tiny golden angle
        rawang=sio.loadmat(fname)['K'] # (512, 1600, 1, 2)
        ang_np=np.squeeze(rawang[:,:,:,0]+1j*rawang[:,:,:,1]).astype(np.complex64)
        ang_np=ang_np[:,200:] # (512, 1400)
        set_ang=np.zeros((Nfr,Nvec,2))
        for idx_fr in range(Nfr):
            set_ang[idx_fr,:,0]=np.real(ang_np[:,idx_fr])*2.*np.pi
            set_ang[idx_fr,:,1]=np.imag(ang_np[:,idx_fr])*2.*np.pi        
        
        # Just for visualization: naive inverse Fourier of undersampled data
        # real_radial_ri # inp: (20, 1400, 512, 2), removed batch dimension
        print('... calculating nufft for full spoke: gold standard')
        inp = real_radial_ri_ts[0,...]
        mynufft_test = Mypnufft_grasp_test(img_size,set_ang.reshape((-1,2)),inp.shape[1],Nvec,Nc,coil,denc)
        real_radial_img_ri=mynufft_test.backward(inp.reshape((-1,2)))
        real_radial_img = (real_radial_img_ri[:,:,0]**2+real_radial_img_ri[:,:,1]**2)**.5 
        real_radial_img -= real_radial_img.min()
        real_radial_img /= real_radial_img.max()
        

        self.mynufft = Mypnufft_grasp(img_size,Nc)
        self.set_ang = set_ang
        self.img_size = img_size
        self.Nc = Nc
        self.Nfr = Nfr
        self.Nvec = Nvec
        self.coil = coil
        self.denc = denc        
        self.real_radial_ri_ts = real_radial_ri_ts
        self.real_radial_img = real_radial_img
