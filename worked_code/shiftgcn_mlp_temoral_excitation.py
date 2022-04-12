"""
temporal attention
from 87.8% to 0.51%
mlp to have every frame importance and excitation every frame
"""

class Long_Term_wsx(nn.Module):
    def __init__(self, in_planes, joint_v=1):
        super(Long_Term_wsx, self).__init__()
        self.in_planes = in_planes * joint_v

        self.pooling = nn.AdaptiveAvgPool2d((None, 1))

        self.long_term = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes // 16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes // 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes // 16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.pooling(x)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        long_term_weight = self.long_term(x)  # n, v*c, t
        long_term_weight = long_term_weight.view(n, v, c, t).permute(0, 2, 3, 1)  # n,v,t,c
        return long_term_weight


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

        self.long_term = Long_Term_wsx(out_channels)

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)

        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = x * self.long_term(x)

        x = self.bn2(x)
        return x