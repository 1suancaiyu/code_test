# 88.4% (+0.6)
class TCE_SENet_two_layer(nn.Module):
    def __init__(self, in_channels, redu=16):
        super(TCE_SENet_two_layer, self).__init__()
        redu_channels = in_channels // redu

        self.spatial_pooling = nn.AdaptiveAvgPool2d((None,1))
        self.relu = nn.ReLU()
        self.conv_squeeze = nn.Conv1d(in_channels=in_channels, out_channels=redu_channels, kernel_size=1)
        self.conv_temporal = nn.Conv1d(in_channels=redu_channels, out_channels=redu_channels, kernel_size=3, padding=1, bias=False, groups=1)
        self.conv_expand = nn.Conv1d(in_channels=redu_channels, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  Temporal_Channel_Excitation_SE")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        x_origin = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        x = self.spatial_pooling(x).squeeze(-1)
        x = self.conv_squeeze(x)
        x = self.conv_temporal(x)
        _,c_r,_ = x.size()
        x = self.relu(x)
        x = x.permute(0,2,1).contiguous().view(n*t,c_r,1)
        x = self.conv_expand(x)
        x_temporal_score = self.sigmoid(x)
        x = x_origin * x_temporal_score + x_origin
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x