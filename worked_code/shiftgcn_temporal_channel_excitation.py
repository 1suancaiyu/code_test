"""
channel attention
temporal channel interaction
from 87.8% to 88.56% (+0.76%)
give temporal info to channel
"""
import torch.nn as nn

# Temporal_Channel_Excitation_conv1d_two_layer
class Temporal_Channel_Excitation_conv1d_two_layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_segment=3):
        super(Temporal_Channel_Excitation_conv1d_two_layer, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduced_channels = self.in_channels // 16

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.tce_tmp_conv = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.tce_expand = nn.Conv1d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)

        # spatial pooling
        x_tce = self.avg_pool(x)

        # reshape each group 3 frame
        x_tce = x_tce.view(n_batch, self.n_segment, c, 1).squeeze(-1).transpose(2, 1).contiguous()
        # temporal conv
        x_tce = self.tce_tmp_conv(x_tce)
        _,c_r,_ = x_tce.size()
        # relu as SEnet
        x_tce = self.relu(x_tce)

        # reshape
        x_tce = x_tce.transpose(2,1).contiguous().view(-1, c_r, 1)

        # 1D convolution, channel expand
        x_tce = self.tce_expand(x_tce)
        # get importance weight for channel dim
        x_tce = self.sigmoid(x_tce)

        # excite channel dim
        x_tce = x_origin * x_tce + x_origin

        # reshape as origin input
        x_tce = x_tce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()

        x_tce = self.bn(x_tce)

        return x_tce