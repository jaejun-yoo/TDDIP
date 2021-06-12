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
from Mypnufft_mc_func_cardiac import *

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.core.debugger import set_trace


class Solver():
    def __init__(self, module, opt):        
        self.opt = opt
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
             
        self.prepare_dataset()        
        
        self.step = 0
        self.t1, self.t2 = None, None
        self.best_psnr, self.best_psnr_step = 0, 0
        self.best_ssim, self.best_ssim_step = 0, 0        
        
        self.net = module.Net(opt).to(self.dev)
        self.net_input_set = get_input_manifold(opt.input_type, opt.num_cycle, self.Nfr).to(self.dev)
        
        p = get_params(opt.opt_over, self.net, self.net_input_set)
        # Compute number of parameters          
        s  = sum([np.prod(list(pnb.size())) for pnb in p]);
        print ('# params: %d' % s)
        
        if opt.input_type.endswith('mapping'):
            self.mapnet = module.MappingNet(opt).to(self.dev)
            print('... adding params of mapping network ...')
            p += self.mapnet.parameters()            
            # Compute number of parameters          
            s  = sum([np.prod(list(pnb.size())) for pnb in p]);
            print ('# params: %d' % s)

        self.optimizer = torch.optim.Adam(p, lr=opt.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=opt.step_size, 
                                                    gamma=opt.gamma)
        self.loss_fn = nn.MSELoss()        
        if opt.isresume is not None:
            self.load(opt.isresume)

    def fit(self):
        opt = self.opt
        self.writer = SummaryWriter(opt.ckpt_root)   
        self.t1 = time.time()
        
        batch_size = opt.batch_size   
        Nc = self.Nc
        Nfr = self.Nfr
        Nvec = self.Nvec
        Nfibo = opt.Nfibo
        coil = self.coil
        denc = self.denc
        net_input_set = self.net_input_set
        syn_radial_ri_ts = self.syn_radial_ri_ts    
        step = self.step       

        while step < opt.max_steps:
            # randomly pick frames to train (batch, default = 1)
            idx_fr=np.random.randint(0, Nfr)
            idx_frs = range(min(idx_fr, Nfr-batch_size), min(idx_fr+batch_size, Nfr))

            net_input_z = self.net_input_set[idx_fr,:].reshape((batch_size,-1))
            net_input_w = self.mapnet(net_input_z)
            net_input_w = net_input_w.reshape((batch_size,1,opt.style_size,opt.style_size)) 
            out_sp = self.net(net_input_w) # e.g., spatial domain output (img) torch.Size([batch_size, 2, 128, 128])
            out_sp = out_sp.permute(0,2,3,1)
            out_kt = []
            gt_kt = []

            for idx_b in range(batch_size):
                idx_fr = idx_frs[idx_b]
                angle = self.set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:] # (3328, 2) 3328 = 13*256
                gt_kt.append(syn_radial_ri_ts[0,:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:,:].reshape(-1,2)) # syn_radial_ri_ts torch.Size([1, 32, 299, 256, 2])
                self.mynufft.X=out_sp[idx_b,:,:,:]
                out_kt.append(self.mynufft(angle,angle.shape[0]//Nvec,Nvec,Nc,coil,denc[:,:angle.shape[0]//Nvec,:]).reshape(-1,2))

            out_kt = torch.cat(out_kt)
            gt_kt = torch.cat(gt_kt)

            total_loss = self.loss_fn(out_kt[...,0],gt_kt[...,0])+ self.loss_fn(out_kt[...,1],gt_kt[...,1])

#             total_loss *= (self.img_size)**2       
            self.optimizer.zero_grad()
            total_loss.backward()   
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar('total loss/training_loss (mse)', total_loss, step)
            
            if (not self.opt.noPLOT) and (step % opt.save_period == 0 or step == opt.max_steps-1):                
                self.evaluate()

            step += 1
            self.step = step
        
        self.writer.close()   
        
    @torch.no_grad()
    def evaluate(self):  
        step = self.step
        max_steps = self.opt.max_steps
        # For visualizations
        # Get Average PSNR and SSIM values for entire frames
        psnr_val_list = []
        ssim_val_list = []
        ims = []
        inp = []
        net_input_w_set = self.mapnet(self.net_input_set).reshape((self.Nfr,1,self.opt.style_size,self.opt.style_size)) 
        for idx_fr in range(self.Nfr):
            tmp_ims = self.net(net_input_w_set[idx_fr:idx_fr+1,...])
            tmp_ims = torch_to_np(tmp_ims).astype('float64') 
            tmp_ims= np.sqrt(tmp_ims[0,:,:]**2+tmp_ims[1,:,:]**2)
            tmp_ims -= tmp_ims.min()
            tmp_ims /= tmp_ims.max()
            syn_radial_img = self.syn_radial_img[:,:,idx_fr]
            gt_cartesian_img = self.gt_cartesian_img[:,:,idx_fr]            
            psnr_val_list += [psnr(gt_cartesian_img, tmp_ims)]
            ssim_val_list += [ssim(gt_cartesian_img, tmp_ims)]  
            net_input_w = torch_to_np(net_input_w_set[idx_fr,:])
            images_grid = np.concatenate([syn_radial_img[None], tmp_ims[None], gt_cartesian_img[None]],axis=2)
            ims.append(images_grid)
            inp.append(net_input_w)
        
        psnr_val = np.array(psnr_val_list).sum()/self.Nfr
        ssim_val = np.array(ssim_val_list).sum()/self.Nfr  
        
        if self.opt.istest:
            print("Saving h5/video (PSNR {:.2f}, SSIM {:.4f} @ {} step)".format(psnr_val, ssim_val, step))
            self.save_video(inp, ims, psnr_val_list, ssim_val_list)
        else:
            self.writer.add_scalar('metrics/psnr', psnr_val, step)
            self.writer.add_scalar('metrics/ssim', ssim_val, step)
            self.writer.add_image('recon_image', ims[1], step)

            self.t2 = time.time()

            if psnr_val >= self.best_psnr:
                self.best_psnr, self.best_psnr_step = psnr_val, step
                self.issave = True
            if ssim_val >= self.best_ssim:
                self.best_ssim, self.best_ssim_step = ssim_val, step
                self.issave = True

            if self.issave:    
                self.save(step)
                self.issave = False

            curr_lr = self.scheduler.get_lr()[0]
            eta = (self.t2-self.t1) * (max_steps-step) /self.opt.save_period / 3600
            print("[{}/{}] {:.2f} {:.4f} (Best PSNR: {:.2f} SSIM {:.4f} @ {} step) LR: {}, ETA: {:.1f} hours"
                .format(step, max_steps, psnr_val, ssim_val, self.best_psnr, self.best_ssim, self.best_psnr_step,
                 curr_lr, eta))

            self.t1 = time.time()
        
        if step == max_steps-1:            
            self.save(step)
            self.save_video(inp, ims, psnr_val_list, ssim_val_list)

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.net_input_set = checkpoint['net_input_set'].to(self.dev)
        self.net.load_state_dict(checkpoint['net_state_dict']) 
        self.net.to(self.dev)
        if self.opt.input_type.endswith('mapping'):
            self.mapnet.load_state_dict(checkpoint['mapnet_state_dict'])
            self.mapnet.to(self.dev)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = checkpoint['scheduler']
        self.step = checkpoint['step']
        self.best_psnr, self.best_psnr_step = checkpoint['best_psnr'], checkpoint['best_psnr_step']
        self.best_ssim, self.best_ssim_step = checkpoint['best_ssim'], checkpoint['best_ssim_step']
        if not self.opt.istest:
            self.step = checkpoint['step']+1        
        
    def save(self, step):        
        print('saving ... ')
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        ckptdict = {
                'step': step,
                'net_input_set': self.net_input_set,
                'net_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                }
        best_scores = {
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_psnr_step': self.best_psnr_step,
                'best_ssim_step': self.best_ssim_step,
                }
        ckptdict = {**ckptdict, **best_scores}
        
        if self.opt.input_type.endswith('mapping'):            
            ckptdict['mapnet_state_dict'] = self.mapnet.state_dict()
                
        torch.save(ckptdict, save_path)
        with open(os.path.join(self.opt.ckpt_root, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)

    def save_video(self, inp, ims, psnr_val_list, ssim_val_list):      
        f = h5py.File(os.path.join(self.opt.ckpt_root, 'final_{}.h5'.format(self.step)),'w')
        f.create_dataset('input_w',data=inp,dtype=np.float32)
        f.create_dataset('data',data=ims,dtype=np.float32)
        f.create_dataset('psnr_val_list',data=psnr_val_list,dtype=np.float32)
        f.create_dataset('ssim_val_list',data=ssim_val_list,dtype=np.float32)
        f.close()
        print('h5 file saved.')
        
        print('creating video, (vmax=0.5).')
        fig = plt.figure(figsize=(10, 10))
        vid = []
        for idx_fr in range(self.Nfr):
            tmp_ims = ims[idx_fr].squeeze()    
            ttl = plt.text(128, -5, idx_fr, horizontalalignment='center', fontsize = 20)
            vid.append([plt.imshow(tmp_ims, animated=True, cmap = 'gray', vmax=0.5),ttl])
        ani = animation.ArtistAnimation(fig, vid, interval=50, blit=True, repeat_delay=1000)        
        ani.save(self.opt.ckpt_root+'/final_video_{}_{}.mp4'.format(os.path.basename(self.opt.ckpt_root),self.step))
        print('video saved')
    
    def prepare_dataset(self):           
        fname = self.opt.fname
        num_cycle = self.opt.num_cycle
        Nfibo = self.opt.Nfibo
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

        # 111.246 degree - golden angle | 23.63 degree - tiny golden angle
        gA=111.246
        one_vec_y=np.linspace(-3.1293208599090576,3.1293208599090576,num=Nvec)[...,np.newaxis]
        one_vec_x=np.zeros((Nvec,1))
        one_vec=np.concatenate((one_vec_y,one_vec_x),axis=1) # (256, 2)

        
        Nang=Nfibo*Nfr
        set_ang=np.zeros((Nang*Nvec,2),np.double) # (995072, 2)
        for i in range(Nang):
            theta=gA*(i+1)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            for j in range(Nvec):
                tmp=np.matmul(R,one_vec[j,:])
                set_ang[i*Nvec+j,0]=tmp[0]
                set_ang[i*Nvec+j,1]=tmp[1]

        data_raw_fname = 'syn_radial_data_cycle%s_Nfibo%s.mat'%(num_cycle, Nfibo)
        # This sampling process takes a bit of time, so we save it once and use it after. 
        
        if os.path.isfile(data_raw_fname):
            data_raw = sio.loadmat(data_raw_fname)['data_raw']
            print('file loaded: %s' % data_raw_fname)
            
        else: 
            data_raw=np.zeros((Nc,Nfibo*Nfr,Nvec)).astype(np.complex64)
            
            # Generate down-sampled data 
            for idx_fr in range(1,Nfr): # Fourier transform per each frame
                print('%s/%s'%(idx_fr,Nfr), '\r', end='')
                angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:]
                mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
                
                tmp=mynufft_test.forward(gt_cartesian_kt_ri[:,idx_fr,:,:,:])
                tmp_c=tmp[...,0]+1j*tmp[...,1]
                tmp_disp=tmp_c.reshape(Nc,Nfibo,Nvec) # (32, 13, 256)

                data_raw[:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:]=tmp_disp

            data_raw=np.transpose(data_raw,(2,1,0))# (256, 299, 32), x-f data
            sio.savemat(data_raw_fname,{'data_raw':data_raw})
            print('file saved: %s' % data_raw_fname)

        
        # Generate down-sampled image
        syn_radial_ri = np.concatenate((np.real(data_raw[...,np.newaxis]),np.imag(data_raw[...,np.newaxis])),axis=3)
        syn_radial_ri = np.transpose(syn_radial_ri,(2,1,0,3)) # (32, 299, 256, 2)
        syn_radial_ri_ts = np_to_torch(syn_radial_ri.astype(np.float32)).cuda().detach() # torch.Size([1, 32, 299, 256, 2]), added batch dimension
        
        # Just for visualization: naive inverse Fourier of undersampled data
        # syn_radial_img
        syn_radial_img_fname = 'syn_radial_img_cycle%s_Nfibo%s.mat'%(num_cycle, Nfibo)
        if os.path.isfile(syn_radial_img_fname):            
            syn_radial_img = sio.loadmat(syn_radial_img_fname)['syn_radial_img']
            print('file loaded: %s' % syn_radial_img_fname)
        else: 
            syn_radial_img=np.zeros((img_size,img_size,Nfr)) # (128, 128, 23)
            print('Get images of the synthetic radial (down-sampled) data')
            for idx_fr in range(Nfr):
                print('%s/%s'%(idx_fr,Nfr), '\r', end='')
                angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:] # (3328, 2)
                inp= torch_to_np(syn_radial_ri_ts[:,:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:,:]) # inp: (32, 13, 256, 2), removed batch dimension

                mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
                gt_re_np=mynufft_test.backward(inp.reshape((-1,2))) # (128, 128, 2)
                syn_radial_img[:,:,idx_fr]=np.sqrt(gt_re_np[:,:,0]**2+gt_re_np[:,:,1]**2) # (128, 128)
                syn_radial_img[:,:,idx_fr] = syn_radial_img[:,:,idx_fr]-syn_radial_img[:,:,idx_fr].min()
                syn_radial_img[:,:,idx_fr] = syn_radial_img[:,:,idx_fr]/syn_radial_img[:,:,idx_fr].max()

            sio.savemat(syn_radial_img_fname,{'syn_radial_img':syn_radial_img})
            print('file saved: %s' % syn_radial_img_fname)

        self.mynufft = Mypnufft_cardiac(img_size,Nc)
        self.set_ang = set_ang
        self.img_size = img_size
        self.Nc = Nc
        self.Nfr = Nfr
        self.Nvec = Nvec
        self.coil = coil
        self.denc = denc
        self.syn_radial_img = syn_radial_img
        self.syn_radial_ri_ts = syn_radial_ri_ts
        self.gt_cartesian_img = gt_cartesian_img