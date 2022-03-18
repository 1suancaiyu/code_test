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
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # shape: (n_batch, c, t, h, w)
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1)) # shape: (n_batch * c, t, 1, 1)
        out = out.view(-1, t) # shape: (n_batch * c, t)
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1) # conv_kernel.shape (n_batch * c, 1, kernel_size, 1)
        local_activation = self.L(out.view(n_batch, c, t)).view(n_batch, c, t, 1, 1) # local_activation.shape (n_batch, c, t, 1, 1)
        new_x = new_x * local_activation # new_x shape: (n_batch, c, t, h, w) local_activation.shape (n_batch, c, t, 1, 1)
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out

# image
tam = TAM(64, n_segment=8, kernel_size=3, stride=1, padding=1)
x = torch.randn([16,300,64,10,10], dtype=torch.float32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
y = tam(x)





"""
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
"""

"""
input:
torch.Size([600, 64, 8])
n_batch, c, t

local
Sequential(
  (0): Conv1d(64, 16, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
  (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv1d(16, 64, kernel_size=(1,), stride=(1,), bias=False)
  (4): Sigmoid()
)
"""