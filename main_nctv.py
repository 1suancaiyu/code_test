
import torch.nn as nn
import torch
import torch.nn.functional as F


class Motion_Temporal_Excitation(nn.Module):
    def __init__(self, in_channels, n_segment=3):
        super(Motion_Temporal_Excitation, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment

        self.reduced_channels = self.in_channels // 16

        self.pad = (0, 0, 0, 0, 0, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        # layers
        self.me_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.me_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.me_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1)
        self.me_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        print('=> Using Motion_Excitation')

    def forward(self, x):
        # get origin
        x_origin = x
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
        me_weight = self.sigmoid(x_me)
        x = x_origin * me_weight # n,c,t,v * n,c,t,1
        return x

class Motion_Temporal_Att(nn.Module):
    def __init__(self, in_channels, n_segment=3):
        super(Motion_Temporal_Att, self).__init__()
        self.spatical_pooling = nn.AdaptiveAvgPool1d(1)
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pad = (0, 0, 0, 1)
    def forward(self, x):
        # get origin
        x_origin = x
        n, c, t, v = x.size()
        x = x.mean(1)
        x_t, _ = x.view(n, t, v).split([t - 1, 1], dim=1)
        _, x_t1 = x.view(n, t, v).split([1, t - 1], dim=1)
        x_motion = x_t1 - x_t
        x_motion = F.pad(x_motion, self.pad, mode="constant", value=0)
        x_motion = x_motion.abs()
        x_motion = self.spatical_pooling(x_motion)
        x_motion = self.temporal_conv(x_motion.squeeze(-1).view(n,1,t))
        x_motion_temporal_score = self.sigmoid(x_motion)
        x_motion_temporal_score = x_motion_temporal_score.view(n, 1, t, 1)
        x = x_origin * x_motion_temporal_score + x_origin
        return x

class Motion_Spatial_Att(nn.Module):
    def __init__(self, in_channels, n_segment=3):
        super(Motion_Spatial_Att, self).__init__()
        self.spatical_pooling = nn.AdaptiveAvgPool1d(1)
        self.spatial_conv = nn.Conv1d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pad = (0, 0, 0, 1)
    def forward(self, x):
        # get origin
        x_origin = x
        n, c, t, v = x.size()
        x = x.mean(1)
        x_t, _ = x.view(n, t, v).split([t - 1, 1], dim=1)
        _, x_t1 = x.view(n, t, v).split([1, t - 1], dim=1)
        x_motion = x_t1 - x_t
        x_motion = F.pad(x_motion, self.pad, mode="constant", value=0)
        x_motion = x_motion.abs()
        x_motion = x_motion.mean(1, keepdim=True)
        x_motion = self.spatial_conv(x_motion)
        x_motion_spatial_score = self.sigmoid(x_motion)
        x_motion_spatial_score = x_motion_spatial_score.unsqueeze(1)
        x = x_origin * x_motion_spatial_score + x_origin
        return x

def main():
    x = torch.randn([30, 64, 300, 25], dtype=torch.float32)
    # module = SCE_SENet(in_channels=64, num_jpts=25)
    module = Motion_Spatial_Att(in_channels=64)
    print(module)
    x = module(x)
    print(x.shape)

if __name__ == "__main__":
    main()