# Code for "TAM: Temporal Adaptive Module for Video Recognition"
# arXiv: 2005.06803
# Zhaoyang liu*, Limin Wang, Wayne Wu, Chen Qian, Tong Lu
# zyliumy@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
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
            nn.Linear(n_segment * 2, kernel_size, bias=False),
            nn.Softmax(-1))

        # input: (n_batch, c, t)
        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        N, T, C, V = x.size()
        x = x.view(N * T, C, V)

        nt, c, v = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, v).permute(0, 2, 1, 3).contiguous() # shape: (n_batch, c, t, v)
        out = self.avg_pool(new_x)
        out = out.view(-1, t) # shape: (n_batch * c, t)
        conv_kernel = self.G(out.view(-1, t))
        conv_kernel = conv_kernel.view(n_batch * c, 1, -1, 1) # conv_kernel.shape (n_batch * c, 1, kernel_size, 1)
        local_activation = self.L(out.view(n_batch, c, t))
        local_activation = local_activation.view(n_batch, c, t, 1) # local_activation.shape (n_batch, c, t, 1)
        new_x = new_x * local_activation # shape: (n_batch, c, t, h, w)
        out = F.conv2d(new_x.view(1, n_batch * c, t, v),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(nt, c, v)

        out = out.view(N, T, C, V)

        return out

# image
# tam = TAM(64, n_segment=8, kernel_size=3, stride=1, padding=1)
# x = torch.zeros([16,300,64,10,10], dtype=torch.float32)
# n, t, c, h, w = x.size()
# x = x.view(n * t, c, h, w)
# y = tam(x)

# skeleton
tam = TAM(64, n_segment=8, kernel_size=3, stride=1, padding=1)
x = torch.zeros([16,300,64,25], dtype=torch.float32)
# n, t, c, v = x.size()
# x = x.view(n * t, c, v)
y = tam(x)


"""
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
"""

