import numpy as np
import torch
import torch.nn as nn 


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MappingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        latent_dim = opt.latent_dim
        style_dim = opt.style_size**2
        hidden_dim = opt.hidden_dim
        depth = opt.depth
        
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.LeakyReLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.LeakyReLU()]
        layers += [nn.Linear(hidden_dim, style_dim)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)        
        return out


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__() 
        inp_ch=opt.input_nch
        ndf=opt.ndf
        out_ch=opt.output_nch
        Nr=opt.Nr
        num_ups=int(np.log2(opt.up_factor))
        need_bias=opt.need_bias
        upsample_mode=opt.upsample_mode
        
        layers = [conv(inp_ch, ndf, 3, bias=need_bias), 
                  nn.InstanceNorm2d(ndf, affine=False),
                  nn.LeakyReLU()]
        
        for _ in range(Nr):
            layers += [conv(ndf, ndf, 3, bias=need_bias),
                       nn.InstanceNorm2d(ndf, affine=False),
                       nn.LeakyReLU()]

        for _ in range(num_ups):
            layers += [nn.Upsample(scale_factor=2, mode=upsample_mode),
                       conv(ndf, ndf, 3, bias=need_bias),                                         
                       nn.InstanceNorm2d(ndf, affine=False),
                       nn.LeakyReLU()]
            for _ in range(Nr):
                layers += [conv(ndf, ndf, 3, bias=need_bias),
                           nn.InstanceNorm2d(ndf, affine=False),
                           nn.LeakyReLU()]

        layers += [conv(ndf, out_ch, 3, bias=need_bias)]


        self.net = nn.Sequential(*layers)
        
    def forward(self, z, s=None):
        out = self.net(z)        
        return out