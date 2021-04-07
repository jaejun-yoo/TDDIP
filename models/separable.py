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
        Nr=opt.Nr
        num_ups=opt.num_ups    
        layer_mask=opt.layer_mask
        num_channels=opt.num_channels
        kernel_size = opt.kernel_size

        #add input layer
        layers = [ nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels[1]),
                    nn.ReLU(True)]
        
        for ii in range(Nr):
            layers += [ nn.Conv2d(num_channels[1], num_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels[1]),
                        nn.ReLU(True)]

        for i in range(num_ups):

            num_channels_in = num_channels[i + 1]
            num_channels_out = num_channels[i + 2]

            if not (layer_mask[i]):

                layers += [ nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_channels_out),
                            nn.ReLU(True)]

                for ii in range(Nr):
                    layers += [ nn.Conv2d(num_channels_out, num_channels_out, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels_out),
                                nn.ReLU(True)]
            else:

                layers += [ nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(num_channels_in, num_channels_out, kernel_size=(kernel_size,1), stride=1, padding=(1,0), bias=False),
                            nn.Conv2d(num_channels_out, num_channels_out, kernel_size=(1,kernel_size), stride=1, padding=(0,1), bias=False),
                            nn.ReLU(True)]

                for ii in range(Nr):
                    layers += [ nn.Conv2d(num_channels_out, num_channels_out, kernel_size=(kernel_size,1), stride=1, padding=(1,0), bias=False),
                                nn.Conv2d(num_channels_out, num_channels_out, kernel_size=(1,kernel_size), stride=1, padding=(0,1), bias=False),
                                nn.BatchNorm2d(num_channels_out),
                                nn.ReLU(True)]

        layers += [ nn.Conv2d(num_channels[-2], num_channels[-1], kernel_size=3, stride=1, padding=1, bias=False)]

        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        out = self.net(z)        
        return out
    