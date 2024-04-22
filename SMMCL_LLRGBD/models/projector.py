import torch
import torch.nn as nn

import math
from timm.models.layers import trunc_normal_  


class projector(nn.Module):
    def __init__(self, in_dim, out_dim=256):   
        super(projector, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(in_dim), 
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(in_dim), 
            nn.Conv2d(in_dim, out_dim, 1))

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

    def forward(self, feature):
        feature = self.project(feature)
        return feature
