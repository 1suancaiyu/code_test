import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            2*self.fold, 2*self.fold,
            kernel_size=3, padding=1, groups=2*self.fold,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

x = torch.zeros([16,8,64,10,10], dtype=torch.float32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
shift = ShiftModule(64, n_segment=8, n_div=2, mode='shift')
y = shift(x)