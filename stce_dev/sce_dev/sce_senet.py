# Spatial_Channel_Excitation 2s-agcn agcn.py 86.2 (-0.3)
class Spatial_Channel_Excitation_SE_mean(nn.Module):
    def __init__(self, in_channels, redu=8):
        super(Spatial_Channel_Excitation_SE_mean, self).__init__()
        redu_channels = in_channels // redu
        self.conv_squeeze = nn.Conv1d(in_channels=in_channels, out_channels=redu_channels, kernel_size=1)
        self.conv_spatial = nn.Conv1d(in_channels=redu_channels, out_channels=redu_channels, kernel_size=1)
        self.conv_expand = nn.Conv1d(in_channels=redu_channels, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  Spatial_Channel_Excitation_SE_mean")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        x_origin = x
        x = x.mean(2)
        x = self.conv_squeeze(x)
        x = self.conv_spatial(x)
        x = x.mean(2, keepdim=True)
        x = self.conv_expand(x)
        x_spatical_score = self.sigmoid(x)
        x_spatical_score = x_spatical_score.unsqueeze(-1)
        x = x_origin * x_spatical_score + x_origin
        return x

# SCE SENet two layers kernel_size = 25 temporal_pooling
class SCE_SENet(nn.Module):
    def __init__(self, in_channels, num_jpts, redu = 16):
        super(SCE_SENet, self).__init__()
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        redu_channels = in_channels // redu
        self.temporal_pooling = nn.AdaptiveAvgPool2d((1,None))
        self.conv_spatial = nn.Conv1d(in_channels, redu_channels, ker_jpt, padding=pad)
        self.relu = nn.ReLU()
        self.conv_expand = nn.Conv1d(redu_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # spatial attention
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