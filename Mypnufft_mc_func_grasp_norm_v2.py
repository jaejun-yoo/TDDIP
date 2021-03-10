import torch
from torch import nn, optim
from torch import Tensor
import numpy as np
import numpy
import scipy.misc
import matplotlib.pyplot
import scipy.io as sio
import nufft

dtype = numpy.complex64

def np_to_torch_(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np).cuda()

def torch_to_np_(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy().astype(np.float32)

class Mypnufft_grasp_func(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing 
    torch.autograd.Function and implementing the forward and backward passes 
    which operate on Tensors.

    NuFFT from https://github.com/jyhmiinlin/pynufft
    """

    @staticmethod
    def forward(ctx,input_r,angle,Nspoke,Nvec,Nc,C,w):
        """
        In the forward pass we receive a Tensor containing the input and return 
        a Tensor containing the output. ctx is a context object that can be used 
        to stash information for backward computation. You can cache arbitrary 
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        N=np.shape(input_r)[-2]
        if np.mod(N,2)==0:
            x=[np.linspace(-N//2,N//2-1,N),
               np.linspace(-N//2,N//2-1,N) 
            ]
        else:
            x=[np.linspace(-N//2,N//2,N),
               np.linspace(-N//2,N//2,N) 
            ]
        
        X = np.meshgrid(x[0], x[1], indexing='ij')
        x1=X[0].reshape(-1)
        x2=X[1].reshape(-1)
        
        ## ctx define
        ctx.x1=x1
        ctx.x2=x2
        ctx.N=N
        ctx.Nc=Nc
        ctx.Nspoke=Nspoke
        ctx.Nvec=Nvec
        ctx.angle=angle
        ctx.wr=w
        ctx.C=C
        
        ###########
        input=torch_to_np_(input_r)
        input_c=input[...,0]+1j*input[...,1]
        
        input_c=np.tile(input_c[np.newaxis],(Nc,1,1))
        input_c*=C
                
        
        y=np.zeros((Nc,Nvec*Nspoke),dtype=np.complex64)
        for it in range(Nc):
            y[it,:] = nufft.nufft2d3(-x1,-x2,input_c[it,:,:].reshape(-1),angle[:,0],angle[:,1],iflag=0)
        # density correction
#         y=y*ctx.wr
        
        
        yr=np.reshape(y,(Nc,Nspoke,Nvec))
        yc=yr*ctx.wr
        
        y=np.reshape(yc,(Nc,Nspoke*Nvec))
        
        y = y[...,np.newaxis]
        y_c = np.concatenate((np.real(y),np.imag(y)),axis=-1)

        y_t = np_to_torch_(y_c.astype(np.float32))
        return y_t

    @staticmethod
    def backward(ctx,grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss 
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        
        angle=ctx.angle
        grad_output_n=torch_to_np_(grad_output)
        
        grad_output=grad_output_n[...,0]+1j*grad_output_n[...,1]
        
        yc=np.reshape(grad_output,(ctx.Nc,ctx.Nspoke,ctx.Nvec))
#         yc=yr*ctx.wr
        
        out=np.zeros((ctx.Nc,ctx.N,ctx.N),dtype=np.complex64)
        
        for it in range(ctx.Nc):
            
            x_re = nufft.nufft2d3(angle[:,0],angle[:,1],yc[it,:,:].reshape(-1),ctx.x1,ctx.x2,iflag=1)
            tmp = x_re.reshape(ctx.N,ctx.N)
            out[it,:,:]=tmp
        
        out=np.sum(out*np.conj(ctx.C),0)/sum( np.abs(ctx.C)**2,0 )

        out*=ctx.N*ctx.N*np.pi
#         out*=ctx.wr.shape[0]*ctx.wr.shape[0]*np.pi
#         out=out*ctx.C.shape[-1]*2*np.pi/ctx.wr.shape[0]
        
        out = out[...,np.newaxis]
        out_c = np.concatenate((np.real(out),np.imag(out)),axis=-1)
        grad_output = np_to_torch_(out_c.astype(np.float32))
        return grad_output, None, None, None, None, None, None

    

class Mypnufft_grasp_test(nn.Module):
    def __init__(self,ImageSize, angle, Nspoke,Nvec,Nc,C,w):
        super(Mypnufft_grasp_test,self).__init__()
        
        
        N=ImageSize
        if np.mod(N,2)==0:
            x=[np.linspace(-N//2,N//2-1,N),
               np.linspace(-N//2,N//2-1,N) 
            ]
        else:
            x=[np.linspace(-N//2,N//2,N),
               np.linspace(-N//2,N//2,N) 
            ]
        X = np.meshgrid(x[0], x[1], indexing='ij')
        x1=X[0].reshape(-1)
        x2=X[1].reshape(-1)
        self.x1=x1
        self.x2=x2
        self.N=N
        self.Nc=Nc
        self.Nspoke=Nspoke
        self.Nvec=Nvec
        self.angle=angle
        self.C=C
        
        wr=np.tile(w[np.newaxis],(self.Nc,self.Nspoke,1))
        self.wr=wr#np.sqrt(wr)
        
    def forward(self,input_r):
        """
        In the forward pass we receive a Tensor containing the input and return 
        a Tensor containing the output. ctx is a context object that can be used 
        to stash information for backward computation. You can cache arbitrary 
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        angle=self.angle
        input=torch_to_np_(input_r)
        input_c=input[...,0]+1j*input[...,1]
        input_c=np.tile(input_c[np.newaxis],(self.Nc,1,1))
        input_c*=self.C

        y=np.zeros((self.Nc,self.Nvec*self.Nspoke),dtype=np.complex64)
        for it in range(self.Nc):
            y[it,:] = nufft.nufft2d3(-self.x1,-self.x2,input_c[it,:,:].reshape(-1),angle[:,0],angle[:,1],iflag=0)
        
        # density correction
        y=y*self.wr.reshape(self.Nc,-1)
        
        
        y = y[...,np.newaxis]
        y_c = np.concatenate((np.real(y),np.imag(y)),axis=-1)

        y_t = np_to_torch_(y_c.astype(np.float32))
        return y_t

    def backward(self,grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss 
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        angle=self.angle
        grad_output_n=torch_to_np_(grad_output)
        grad_output=grad_output_n[...,0]+1j*grad_output_n[...,1]
        
        yc=np.reshape(grad_output,(self.Nc,self.Nspoke,self.Nvec))
#         yc=yr*self.wr
        
        out=np.zeros((self.Nc,self.N,self.N),dtype=np.complex64)
        for it in range(self.Nc):
            x_re = nufft.nufft2d3(angle[:,0],angle[:,1],yc[it,:,:].reshape(-1),self.x1,self.x2,iflag=1)
            tmp = x_re.reshape(self.N,self.N)
            out[it,:,:]=tmp
       
        #print(out.shape)
        #out*=(np.pi/2/self.Nspoke)
        # coil combination
        out=np.sum(out*np.conj(self.C),0)/sum( np.abs(self.C)**2,0 )
        
        out*=self.N*self.N*np.pi
        
        out = out[...,np.newaxis]
        out_c = np.concatenate((np.real(out),np.imag(out)),axis=-1)
        grad_output = np_to_torch_(out_c)
        return grad_output
    
    
class Mypnufft_grasp(nn.Module):
    def __init__(self,ImageSize,Nc):
        super(Mypnufft_grasp,self).__init__()
        
        self.X=Tensor(ImageSize,ImageSize,Nc).fill_(0).cuda()
        
    def forward(self,angles, Nspoke,Nvec,Nc,C,w):
        
        return  Mypnufft_grasp_func.apply(self.X, angles, Nspoke,Nvec,Nc,C,w)

    