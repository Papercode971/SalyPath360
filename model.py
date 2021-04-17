import torch
from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from kornia.geometry import SpatialSoftArgmax2d

class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)
        
        
    def forward(self, x):
        x = self.pool(x)
        return x


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M_2':
            layers += [Downsample(kernel_size= 2)]
        elif v == 'M_4':
            layers += [Downsample(kernel_size= 4)]
        else:
            conv = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv, ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'global_attention': [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 'M_4', 512, 512, 512],
    'based_AM'        : [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 512, 512, 512]
      }
#global_attention = make_conv_layers(cfg['global_attention'])
based_AM = make_conv_layers(cfg['based_AM'])

# create unpooling layer 
class Upsample(nn.Module):
    # specify the scale_factor for upsampling 
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


# reshape vectors layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Scanpath_based_Attention_module(nn.Module):


    """
    not yet
    """

    def  __init__(self):
        super(Scanpath_based_Attention_module,self).__init__()
        Based_Attention_Module = based_AM
        soft_sam = SpatialSoftArgmax2d(normalized_coordinates=False) 
        self.soft_sam = soft_sam 
        self.encoder = torch.nn.Sequential(*Based_Attention_Module)
        self.attention_module = torch.nn.Sequential(*[
            Downsample(kernel_size= 2),
            Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size= 2),
            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
            Upsample(scale_factor=4 , mode='nearest' )
        ])


        decoder_list=[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU()
            #Upsample(scale_factor=2, mode='nearest'),
            
        ]


        decoder_list_hm =[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        self.decoder_hm = torch.nn.Sequential(*decoder_list_hm)

        self.decoder = torch.nn.Sequential(*decoder_list)
        self.aux = torch.nn.Sequential(*[
            #Upsample(scale_factor=2, mode='nearest'),
            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU()
        ])     


       
        for name, param in self.aux.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 150.0)

        print("Model initialized, Sal_based_Attention_module")
    

    
    def set_freez(self,epoch):
        if epoch < 3:
          for name, param in self.named_parameters():
              #if 'decoder' in name or 'aux' in name or 'decoder_hm' in name :
              if 'decoder_hm' in name :
                  param.requires_grad = True
              else :
                  param.requires_grad = False

        else:        
          for name, param in self.named_parameters():
              param.requires_grad = True


    def forward(self, input):       
        x = self.encoder(input)  
        x,y = self.shared_attention(x)     
        hm = self.decoder_hm(x)
        x = self.decoder(x)
        x = self.aux(x)
        #a= torch.tensor([[1/641,1/321]]).repeat(100,1).unsqueeze(0)
        return self.soft_sam(x),hm,y 

    def shared_attention(self,input):
        y = self.attention_module(input)
        repeted  = y.repeat(1,512,1,1)
        product  = input*repeted
        added    = input+product
        
        return added,y 
