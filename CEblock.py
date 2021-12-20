import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

# skeleton CE model
class CEModel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_segment=3):
        super(CEModel, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduced_channels = self.in_channels // 16
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.action_p2_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.action_p2_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1, stride=1)
        self.action_p2_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=self.stride)

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x


        # 2D convolution: c*T*1*1, channel excitation
        x_ce = self.avg_pool(x)
        x_ce = self.action_p2_squeeze(x_ce)
        n, c_r, t, v = x_ce.size()
        x_ce = x_ce.view(n_batch, self.n_segment, c_r, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        x_ce = self.action_p2_conv1(x_ce)
        x_ce = self.relu(x_ce)

        # reshape
        x_ce = x_ce.transpose(2,1).contiguous().view(-1, c_r, 1)
        x_ce = x_ce.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()

        # expand
        x_ce = self.action_p2_expand(x_ce)

        # ending
        x_ce = self.sigmoid(x_ce)
        x_ce = x_origin * x_ce + x_origin
        x_ce = self.action_p2_out(x_ce)

        return x_ce

# image test
'''
inception_3a_1x1 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

x = torch.zeros([16,300,64,10,10], dtype=torch.float32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
ce = CEModel(inception_3a_1x1)
y = ce(x)
print(y.shape)
'''

# skeleton test


x = torch.zeros([16,64,300,25], dtype=torch.float32)
n, c, t, v = x.size()
ce = CEModel(64,64,stride=2)
y = ce(x)
print(y.shape)