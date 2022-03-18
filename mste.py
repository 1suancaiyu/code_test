import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import random

def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)

def mlp_sigmoid(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, in_planes//16, kernel_size, **kwargs),
                            nn.BatchNorm1d(in_planes//16),
                            nn.LeakyReLU(inplace=True),
                            conv1d(in_planes//16, out_planes, kernel_size, **kwargs),
                            nn.Sigmoid())

def conv_bn(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, out_planes, kernel_size, **kwargs),
                            nn.BatchNorm1d(out_planes))


class MSTE(nn.Module):
    def __init__(self, in_planes, out_planes, part_num):
        super(MSTE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num

        self.score = mlp_sigmoid(in_planes*part_num, in_planes*part_num, 1, groups=part_num)

        self.short_term = nn.ModuleList([conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num),
                                conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num)])

    def get_frame_level(self, x):
        return x

    def get_short_term(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(n, -1, s)
        temp = self.short_term[0](x)
        short_term_feature = temp + self.short_term[1](temp)
        return short_term_feature.view(n, h, c, s).permute(0, 3, 2, 1).contiguous()

    def get_long_term(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        pred_score = self.score(x.view(n, h * c, s)).view(n, h, c, s)
        long_term_feature = x.mul(pred_score).sum(-1).div(pred_score.sum(-1))
        long_term_feature = long_term_feature.unsqueeze(1).repeat(1, s, 1, 1)
        return long_term_feature.permute(0, 1, 3, 2).contiguous()

    def forward(self, x):
        multi_scale_feature = [self.get_frame_level(x), self.get_short_term(x), self.get_long_term(x)]
        return multi_scale_feature


# n, s, c, h  n: batch_size; C: channel;
x = torch.randn([16,60,128,25], dtype=torch.float32)
multi_scale = MSTE(in_planes=128, out_planes=128, part_num=25)
t_f, t_s, t_l = multi_scale(x)






"""
short term
n, h*c, t
torch.Size([16, 3200, 60])

ModuleList(
  (0): Sequential(
    (0): Conv1d(3200, 3200, kernel_size=(3,), stride=(1,), padding=(1,), groups=25, bias=False)
    (1): BatchNorm1d(3200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): Sequential(
    (0): Conv1d(3200, 3200, kernel_size=(3,), stride=(1,), padding=(1,), groups=25, bias=False)
    (1): BatchNorm1d(3200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)


long term
n, h*c, t
torch.Size([16, 3200, 60])

mlp_sigmoid
Sequential(
  (0): Conv1d(3200, 200, kernel_size=(1,), stride=(1,), groups=25, bias=False)
  (1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): LeakyReLU(negative_slope=0.01, inplace=True)
  (3): Conv1d(200, 3200, kernel_size=(1,), stride=(1,), groups=25, bias=False)
  (4): Sigmoid()
)

"""