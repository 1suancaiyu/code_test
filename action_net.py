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

        # shifting
        self.action_shift = nn.Conv1d(
            self.in_channels, self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels,
            bias=False)
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1  # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right

        if 2 * self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1  # fixed

        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), bias=False, padding=(1, 1, 1))

        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
                                         bias=False, padding=1,
                                         groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))

        # motion excitation
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3),
                                         stride=(1, 1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))
        print('=> Using ACTION')

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x_shift = x

        # 2D convolution: c*T*1*1, channel excitation
        x_p2 = self.avg_pool(x_shift)
        x_p2 = self.action_p2_squeeze(x_p2)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2, 1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        x_p2 = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2 + x_shift


        return x_p2


inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))

x = torch.zeros([16,300,192,10,10], dtype=torch.float32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
act = Action(inception_3a_1x1)
act(x)