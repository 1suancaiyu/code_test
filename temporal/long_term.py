import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import random

class Long_Term(nn.Module):
    def __init__(self, in_planes, part_num):
        super(Long_Term, self).__init__()
        self.in_planes = in_planes*part_num

        self.lt = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes//16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes//16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        n, t, c, v = x.size()
        long_term_weight = self.lt(x.permute(0, 3, 2, 1).contiguous().view(n, v * c, t)) # n, v*c, t
        long_term_weight = long_term_weight.view(n, v, c, t).permute(0, 3, 2, 1) # n,v,c,t
        x = x + x * long_term_weight
        return x


# n, s, c, h  n: batch_size; C: channel; s: frame num; h: features;
x = torch.rand([16,60,64,25], dtype=torch.float32)
lt = Long_Term(in_planes=64, part_num=25)
t_l = lt(x)

