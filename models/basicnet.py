import torch
import torch.nn as nn 

class MappingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        latent_dim = opt.latent_dim
        style_dim= opt.style_dim
        depth = opt.depth
        
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        layers += [nn.Linear(512, style_dim)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)        
        return out
    
class Net(nn.Module):
    def __init__(self, opt):
        super().__init__() 
        inp=opt.input_depth
        ndf=opt.ndf
        out_ch=opt.num_output_ch
        Nr=opt.Nr
        num_ups=opt.num_ups
        need_tanh=opt.need_tanh
        need_bias=opt.need_bias
        upsample_mode=opt.upsample_mode
        need_convT = opt.need_convT

        layers = [ nn.Conv2d(inp, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(True)]
        for ii in range(Nr):
            layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                        nn.BatchNorm2d(ndf),
                        nn.ReLU(True)]

        for i in range(num_ups-1):
            if need_convT:
                layers += [ nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=need_bias),
                            nn.BatchNorm2d(ndf),
                            nn.ReLU(True)]

                for ii in range(Nr):
                    layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                                nn.BatchNorm2d(ndf),
                                nn.ReLU(True)]

            else:
                layers += [ nn.Upsample(scale_factor=2, mode=upsample_mode),
                            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                            nn.BatchNorm2d(ndf),
                            nn.ReLU(True)]
                for ii in range(Nr):
                    layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                                nn.BatchNorm2d(ndf),
                                nn.ReLU(True)]

        if need_convT:
            layers += [nn.ConvTranspose2d(ndf, ndf , 4, 2, 1, bias=need_bias),
                            nn.BatchNorm2d(ndf),
                            nn.ReLU(True)]

            for ii in range(Nr):
                layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                            nn.BatchNorm2d(ndf),
                            nn.ReLU(True)]
            layers += [ nn.Conv2d(ndf, out_ch, kernel_size=3, stride=1, padding=1, bias=need_bias)]  
        else:
            layers += [nn.Upsample(scale_factor=2, mode=upsample_mode),
                           nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                           nn.BatchNorm2d(ndf),
                           nn.ReLU(True)]

            for ii in range(Nr):
                layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                            nn.BatchNorm2d(ndf),
                            nn.ReLU(True)]
            layers += [ nn.Conv2d(ndf, out_ch, kernel_size=3, stride=1, padding=1, bias=need_bias)]

        if need_tanh:
            layers += [ nn.Tanh(),]

        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        out = self.net(z)        
        return out
    
# def Net(opt):    
#     inp=opt.input_depth
#     ndf=opt.ndf
#     out_ch=opt.num_output_ch
#     Nr=opt.Nr
#     num_ups=opt.num_ups
#     need_tanh=opt.need_tanh
#     need_bias=opt.need_bias
#     upsample_mode=opt.upsample_mode
#     need_convT = opt.need_convT

#     layers = [ nn.Conv2d(inp, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                 nn.BatchNorm2d(ndf),
#                 nn.ReLU(True)]
#     for ii in range(Nr):
#         layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                     nn.BatchNorm2d(ndf),
#                     nn.ReLU(True)]
    
#     for i in range(num_ups-1):
#         if need_convT:
#             layers += [ nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=need_bias),
#                         nn.BatchNorm2d(ndf),
#                         nn.ReLU(True)]

#             for ii in range(Nr):
#                 layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                             nn.BatchNorm2d(ndf),
#                             nn.ReLU(True)]
            
#         else:
#             layers += [ nn.Upsample(scale_factor=2, mode=upsample_mode),
#                         nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                         nn.BatchNorm2d(ndf),
#                         nn.ReLU(True)]
#             for ii in range(Nr):
#                 layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                             nn.BatchNorm2d(ndf),
#                             nn.ReLU(True)]
           
#     if need_convT:
#         layers += [nn.ConvTranspose2d(ndf, ndf , 4, 2, 1, bias=need_bias),
#                         nn.BatchNorm2d(ndf),
#                         nn.ReLU(True)]

#         for ii in range(Nr):
#             layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                         nn.BatchNorm2d(ndf),
#                         nn.ReLU(True)]
#         layers += [ nn.Conv2d(ndf, out_ch, kernel_size=3, stride=1, padding=1, bias=need_bias)]  
#     else:
#         layers += [nn.Upsample(scale_factor=2, mode=upsample_mode),
#                        nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                        nn.BatchNorm2d(ndf),
#                        nn.ReLU(True)]

#         for ii in range(Nr):
#             layers += [ nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
#                         nn.BatchNorm2d(ndf),
#                         nn.ReLU(True)]
#         layers += [ nn.Conv2d(ndf, out_ch, kernel_size=3, stride=1, padding=1, bias=need_bias)]
    
#     if need_tanh:
#         layers += [ nn.Tanh(),]

#     model =nn.Sequential(*layers)
#     return model

