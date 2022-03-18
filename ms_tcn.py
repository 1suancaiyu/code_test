import torch.nn as nn
import torch.nn.functional as F
import torch

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class Motion_Excitation(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment=3):
        super(Motion_Excitation, self).__init__()
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
        n, c, t, v = x.size()

        # get n_batch
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
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
        x = x_plus1 - x_plus0  # torch.Size([2000, 2, 4, 25]) torch.Size([2000, 2, 4, 25])

        # pading
        x = F.pad(x, self.pad, mode="constant", value=0)  # torch.Size([2000, 2, 4, 25]) -> orch.Size([2001, 2, 4, 25])

        # spatical pooling
        x = x.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()
        x = self.avg_pool(x)

        # expand
        x = self.me_expand(x)  # torch.Size([6000, 64, 1])

        # sigmoid
        x = x.permute(0, 2, 1, 3).contiguous().view(n*t, c, 1)
        x_me_score = self.sigmoid(x)

        x = x_origin * x_me_score

        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x

class Spatical_Channel_Excitation(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment=3):
        super(Spatical_Channel_Excitation, self).__init__()
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
        n, c, t, v = x.size()
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        x = x.mean(1, keepdim=True) # nt,1,v
        x = gcn(x)
        x_spatical_score = nn.Sigmoid(x)
        x = x_origin * x_spatical_score
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        return x

class Channel_Excitation(nn.Module):
    def __init__(self, out_channels, kernel_size, rd):
        super(Channel_Excitation, self).__init__()

        pad = int((kernel_size - 1) / 2)

        self.ce = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(out_channels,
                      out_channels // rd,
                      kernel_size=(kernel_size, 1),  # short_term_feature
                      stride=1,
                      bias=False,
                      padding=(pad,0),
                      groups=1),
            nn.BatchNorm2d(out_channels // rd),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // rd, out_channels, 1, bias=False),
            nn.Sigmoid())

        print('=> Using Channel_Excitation')

    def forward(self, x):
        x_ce_score = self.ce(x)
        x = x * x_ce_score # (n,c,t,v) * (n,c,t,1)
        return x

class Long_Term_Excitation(nn.Module):
    def __init__(self, in_planes, joint_v=1):
        super(Long_Term_Excitation, self).__init__()
        self.in_planes = in_planes * joint_v

        self.pooling = nn.AdaptiveAvgPool2d((None, 1))

        self.long_term = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes // 16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes // 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes // 16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

        print('=> Using Long_Term_Excitation')

    def forward(self, x):
        x_origin = x
        x = self.pooling(x)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        long_term_weight = self.long_term(x)  # n, v*c, t

        # mean all the frames get the longterm features and add to each frames
        long_term_weight = long_term_weight.mean(-1).unsqueeze(-1).repeat(1,1,t) + long_term_weight

        # excitation
        long_term_weight = long_term_weight.view(n, v, c, t).permute(0, 2, 3, 1)  # n,t,c,v
        x = x_origin * long_term_weight
        return x

class Long_Term_Channel_Excitation(nn.Module):
    def __init__(self, in_planes):
        super(Long_Term_Channel_Excitation, self).__init__()
        self.in_planes = in_planes

        self.pooling = nn.AdaptiveAvgPool2d((None, 1))

        self.long_term = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes // 16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes // 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes // 16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

        print('=> Using Long_Term_Channel_Excitation')

    def forward(self, x):
        x_origin = x
        # x = self.pooling(x)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n*v, c, t)
        long_term_weight = self.long_term(x)  # n*v, c, t

        # mean all the frames get the longterm features and add to each frames
        long_term_weight = long_term_weight.mean(-1).unsqueeze(-1)

        x = x * long_term_weight

        # excitation
        x = x.view(n, v, c, t).permute(0, 2, 3, 1)  # n,t,c,v
        return x

class unit_ctrgcn_tcn(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1):
        super(unit_ctrgcn_tcn, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_stgcn_tcn(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, dropout=0):
        super(unit_stgcn_tcn, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        return self.tcn(x)

class multi_scale_unit_tcn(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(multi_scale_unit_tcn, self).__init__()

        # self.tcn = unit_stgcn_tcn(out_channels, kernel_size, stride)
        self.tcn = unit_ctrgcn_tcn(out_channels, kernel_size, stride)

        self.me = Motion_Excitation(out_channels, out_channels, n_segment=3)
        self.ce = Channel_Excitation(out_channels, kernel_size=5, rd=4)
        self.lt = Long_Term_Excitation(out_channels)

        self.lct = Long_Term_Channel_Excitation(out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.tcn(x)

        x_ce = x + self.ce(x)
        x_me = x + self.me(x)
        x_lt = x + self.lt(x)
        x_lct = x + self.lct(x)

        x = self.bn( x_lct )
        # x = self.bn( x_me + x_ce + x_lt )
        return x

def main():
    x = torch.randn([30, 64, 300, 25], dtype=torch.float32)
    mtm = multi_scale_unit_tcn(out_channels=64, kernel_size=5, stride=2)
    x = mtm(x)
    print(x.shape)

if __name__ == "__main__":
    main()