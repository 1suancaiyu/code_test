# SCE SENet two layers kernel_size = 25 temporal_pooling
# 2s-agcn agcn rise 1.3%
class SCE(nn.Module):
    def __init__(self, in_channels, num_jpts=25, redu = 16):
        super(SCE, self).__init__()
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        redu_channels = in_channels // redu
        self.temporal_pooling = nn.AdaptiveAvgPool2d((1,None))
        self.conv_spatial = nn.Conv1d(in_channels, redu_channels, ker_jpt, padding=pad)
        self.relu = nn.ReLU()
        self.conv_expand = nn.Conv1d(redu_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>>>>>>>>>>>>>  SCE")
    def forward(self, x):
        x_origin = x
        x = self.temporal_pooling(x)
        x = x.squeeze(-2)
        x = self.conv_spatial(x)
        x = self.relu(x)
        x = self.conv_expand(x)
        x = x.unsqueeze(-2)
        sce_weights = self.sigmoid(x)
        x = x_origin * sce_weights + x_origin
        return x

# TCE temporal pooling
# 2sagcn agcn rise 0.79
class TCE(nn.Module):
    def __init__(self, in_channels, redu=16):
        super(TCE, self).__init__()
        redu_channels = in_channels // redu

        self.spatial_pooling = nn.AdaptiveAvgPool2d((None,1))
        self.relu = nn.ReLU()
        self.conv_temporal = nn.Conv1d(in_channels=in_channels, out_channels=redu_channels, kernel_size=3, padding=1, bias=False, groups=1)
        self.conv_expand = nn.Conv1d(in_channels=redu_channels, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channels)
        print(">>>>>>  TCE")

    def forward(self, x):
        x_origin = x
        x = self.spatial_pooling(x)
        x = x.squeeze(-1)
        x = self.conv_temporal(x)
        _,c_r,_ = x.size()
        x = self.relu(x)
        x = self.conv_expand(x)
        x = x.unsqueeze(-1)
        x_temporal_score = self.sigmoid(x)
        x = x_origin * x_temporal_score + x_origin
        return x