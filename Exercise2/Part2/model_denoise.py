
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm2d

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # If after the conv2d there is a batchNorm you want to put bias = False
        # The batchnorm has a learnable bias, so you don't want to have a bias that goes into it
        use_bias = False

        # First block
        self.layer1 = Conv2d(1, 64, 3, padding = 1, bias = use_bias)
        self.layer2 = BatchNorm2d(64)
        self.layer3 = ReLU(inplace=True)
        
        #It works like a list in pytorch
        self.central_layers = nn.ModuleList()
        central_layers_blocks = 2
        
        for i in range(central_layers_blocks):
            self.central_layers.append(Conv2d(64, 64, 3, padding = 1, bias = use_bias))
            self.central_layers.append(BatchNorm2d(64))
            self.central_layers.append(ReLU(inplace=True))

            # Append to self.central_layers conv2d, batch norm and relu as many times as you want
        
        use_bias = True                   
        self.last_layer = Conv2d(64, 1, 3, padding = 1, bias = use_bias)
        
    def forward(self,x):
                               
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        for layer in self.central_layers:
            out = layer(out)
                
        out = self.last_layer(out)
        
        return out + x