import torch
import torch.nn as nn
import torch.nn.functional as F    

import math
from timm.models.layers import trunc_normal_  


class CrossModu(nn.Module):
    def __init__(self, dim, s=.5, c=.5):  
        super(CrossModu, self).__init__()

        self.s = s
        self.c = c 
        self.Spatial = SpatialCoes(dim)
        self.Channel = ChannelCoes(dim)

        self.apply(self._init_weights)  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, color, other):   
        SpaCoes = self.Spatial(color, other) 
        ChaCoes = self.Channel(color, other) 
        color_out = color + other*SpaCoes*self.s + other*ChaCoes*self.c  # Features are modulated with a factor of 0.5 following CMX.
        other_out = other + color*SpaCoes*self.s + color*ChaCoes*self.c
        return color_out, other_out 


class SpatialCoes(nn.Module):
    def __init__(self, dim):
        super(SpatialCoes, self).__init__()
        self.dim = dim   
        self.encode = nn.Sequential(
                    nn.Conv2d(2*self.dim, 2*self.dim, 1),
                    nn.GELU(),
                    nn.Conv2d(2*self.dim, 2*self.dim, 1),
                    nn.GELU(),
                    nn.Conv2d(2*self.dim, 1, 1), 
                    nn.Sigmoid())  

        self.apply(self._init_weights)  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        coes = self.encode(x) 
        return coes


class ChannelCoes(nn.Module):
    def __init__(self, dim):
        super(ChannelCoes, self).__init__()
        self.dim = dim  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.encode = nn.Sequential(               
                    nn.Conv2d(4*self.dim, 4*self.dim, 1),
                    nn.GELU(),
                    nn.Conv2d(4*self.dim, 4*self.dim, 1),
                    nn.GELU(),
                    nn.Conv2d(4*self.dim, self.dim, 1), 
                    nn.Sigmoid())

        self.apply(self._init_weights)   

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        coes = torch.cat((avg, max), dim=1) 
        coes = self.encode(coes) 
        return coes


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConv, self).__init__()
        self.fusion_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

        self.apply(self._init_weights)  
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        fusion = self.fusion_conv(torch.cat([x,y], dim=1))
        return fusion
