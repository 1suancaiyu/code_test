import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels // 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1))

        print('=> Using ACTION STE')

    def forward(self, x):
        nt, c, v = x.size()  # x.shape torch.Size([n*8, 64, 56, 56])
        n_batch = nt // self.n_segment
        x_shift = x

        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, c, v = x_shift.size()
        x_p1 = x_shift.view(n_batch, self.n_segment, c, v).transpose(2, 1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1) # 这样对吗？应该GCN+TCN吗？
        x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, v)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift

        return x_p1


inception_3a_1x1 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
act = Action(inception_3a_1x1)

"""
# images input
image = torch.zeros([16,30,192,10,10], dtype=torch.float32)
n, t, c, h, w = image.size()
image = image.view(n * t, c, h, w)
act(image)

"""

# skeleton input
ske = torch.zeros([16,300,64,25], dtype=torch.float32)
n, t, c, v = ske.size()
ske = ske.view(n * t, c, v)

act(ske)
