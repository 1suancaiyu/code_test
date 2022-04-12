import torch.nn as nn
import torch.nn.functional as F

class Motion_Excitation_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment=3):
        super(Motion_Excitation_TCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_segment = n_segment

        self.reduced_channels = self.in_channels // 16

        self.pad = (0, 0, 0, 0, 0, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        # layers
        self.me_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.me_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.me_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1)
        self.me_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        print('=> Using Motion_Excitation')

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()

        # get n_batch
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        nt, c, v = x.size()
        n_batch = nt // self.n_segment

        # squeeze conv
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        x = self.me_squeeze(x)
        x = self.me_bn1(x)
        n, c_r, t, v = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c_r, v)

        # temporal split
        nt, c_r, v = x.size()
        x_plus0, _ = x.view(n_batch, self.n_segment, c_r, v).split([self.n_segment - 1, 1], dim=1)  # x(t) torch.Size([2000, 2, 4, 25])

        # x(t+1) conv
        x = x.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()
        x_plus1 = self.me_conv1(x)
        x_plus1 = x_plus1.permute(0, 2, 1, 3).contiguous().view(n * t, c_r, v)
        _, x_plus1 = x_plus1.view(n_batch, self.n_segment, c_r, v).split([1, self.n_segment - 1], dim=1)  # x(t+1) torch.Size([2000, 2, 4, 25])

        # subtract
        x_me = x_plus1 - x_plus0  # torch.Size([2000, 2, 4, 25]) torch.Size([2000, 2, 4, 25])

        # pading
        x_me = F.pad(x_me, self.pad, mode="constant", value=0)  # torch.Size([2000, 2, 4, 25]) -> orch.Size([2001, 2, 4, 25])

        # spatical pooling
        x_me = x_me.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()
        x_me = self.avg_pool(x_me)

        # expand
        x_me = self.me_expand(x_me)  # torch.Size([6000, 64, 1])

        # sigmoid
        x_me = self.sigmoid(x_me)

        return x_me