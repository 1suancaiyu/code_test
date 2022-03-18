# Code for "TAM: Temporal Adaptive Module for Video Recognition"
# arXiv: 2005.06803
# Zhaoyang liu*, Limin Wang, Wayne Wu, Chen Qian, Tong Lu
# zyliumy@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class long_term_wsx(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(long_term_wsx, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        print('TAM with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, n_segment, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)

        nt, c, v = x.size()
        t = self.n_segment
        n_batch = nt // t
        x = new_x = x.view(n_batch, t, c, v).permute(0, 2, 1, 3).contiguous() # shape: (n_batch, c, t, v)
        new_x = self.avg_pool(new_x)
        new_x = new_x.view(-1, t) # shape: (n_batch * c, t)
        long_term_score = self.G(new_x.view(-1, t))

        long_term_score = long_term_score.view(n_batch, c, t)
        long_term_score = long_term_score.unsqueeze(-1)
        long_term_score = long_term_score.repeat(1,1,1,v)
        x = x.mul(long_term_score)

        return x

# skeleton
tam = Global_TAM_Skeleton(64, n_segment=300, kernel_size=3, stride=1, padding=1)
x = torch.randn([16,64,300,25], dtype=torch.float32)
y = tam(x)


"""
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
"""

