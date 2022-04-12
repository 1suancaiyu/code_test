
import torch.nn as nn
import torch

# class Temporal_Channel_Excitation_conv1d_two_layer(nn.Module):
#     def __init__(self, in_channels, n_segment=3, redu=8):
#         super(Temporal_Channel_Excitation_conv1d_two_layer, self).__init__()
#         self.n_segment = n_segment
#         self.in_channels = in_channels
#         self.reduced_channels = self.in_channels // redu
#
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#         self.tce_tmp_conv = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.tce_expand = nn.Conv1d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)
#         self.bn = nn.BatchNorm2d(self.in_channels)
#
#     def forward(self, x):
#         # get origin
#         n, c, t, v = x.size()
#         n_batch = n * t // self.n_segment
#         x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
#
#         # spatial pooling
#         x_tce = self.avg_pool(x)
#
#         # reshape each group 3 frame
#         x_tce = x_tce.view(n_batch, self.n_segment, c, 1).squeeze(-1).transpose(2, 1).contiguous()
#         # temporal conv
#         x_tce = self.tce_tmp_conv(x_tce)
#         _,c_r,_ = x_tce.size()
#         # relu as SEnet
#         x_tce = self.relu(x_tce)
#
#         # reshape
#         x_tce = x_tce.transpose(2,1).contiguous().view(-1, c_r, 1)
#
#         # 1D convolution, channel expand
#         x_tce = self.tce_expand(x_tce)
#         # get importance weight for channel dim
#         x_tce = self.sigmoid(x_tce)
#
#         # excite channel dim
#         x_tce = x_origin * x_tce + x_origin
#
#         # reshape as origin input
#         x_tce = x_tce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
#
#         x_tce = self.bn(x_tce)
#
#         return x_tce
#
# # Spatial_Channel_Excitation
# class Spatial_Channel_Excitation_SE(nn.Module):
#     def __init__(self, in_channels, redu=8):
#         super(Spatial_Channel_Excitation_SE, self).__init__()
#         redu_channels = in_channels // redu
#         self.conv_squeeze = nn.Conv1d(in_channels=in_channels, out_channels=redu_channels, kernel_size=1)
#         self.conv_spatial = nn.Conv1d(in_channels=redu_channels, out_channels=redu_channels, kernel_size=1)
#         self.conv_expand = nn.Conv1d(in_channels=redu_channels, out_channels=in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         print(">>>>>>  Spatial_Channel_Excitation")
#
#     def forward(self, x):
#         # get origin
#         n, c, t, v = x.size()
#         x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
#         x = self.conv_squeeze(x)
#         x = self.conv_spatial(x)
#         x = self.conv_expand(x)
#         x_spatical_score = self.sigmoid(x)
#         x = x_origin * x_spatical_score
#         x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
#         return x


# MTA_STCE channel split
class MTA_STCE(nn.Module):
    def __init__(self, in_channels):
        super(MTA_STCE, self).__init__()
        assert in_channels % 2 == 0
        self.width = in_channels // 2

        class Temporal_Channel_Excitation_conv1d_two_layer(nn.Module):
            def __init__(self, in_channels, n_segment=3, redu=8):
                super(Temporal_Channel_Excitation_conv1d_two_layer, self).__init__()
                self.n_segment = n_segment
                self.in_channels = in_channels
                self.reduced_channels = self.in_channels // redu

                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.relu = nn.ReLU(inplace=True)
                self.sigmoid = nn.Sigmoid()

                self.tce_tmp_conv = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=3, stride=1,
                                              bias=False, padding=1, groups=1)
                self.tce_expand = nn.Conv1d(in_channels=self.reduced_channels, out_channels=self.in_channels,
                                            kernel_size=1)
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
                _, c_r, _ = x_tce.size()
                # relu as SEnet
                x_tce = self.relu(x_tce)

                # reshape
                x_tce = x_tce.transpose(2, 1).contiguous().view(-1, c_r, 1)

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

        # Spatial_Channel_Excitation
        class Spatial_Channel_Excitation_SE(nn.Module):
            def __init__(self, in_channels, redu=8):
                super(Spatial_Channel_Excitation_SE, self).__init__()
                redu_channels = in_channels // redu
                self.conv_squeeze = nn.Conv1d(in_channels=in_channels, out_channels=redu_channels, kernel_size=1)
                self.conv_spatial = nn.Conv1d(in_channels=redu_channels, out_channels=redu_channels, kernel_size=1)
                self.conv_expand = nn.Conv1d(in_channels=redu_channels, out_channels=in_channels, kernel_size=1)
                self.sigmoid = nn.Sigmoid()
                print(">>>>>>  Spatial_Channel_Excitation")

            def forward(self, x):
                # get origin
                n, c, t, v = x.size()
                x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
                x = self.conv_squeeze(x)
                x = self.conv_spatial(x)
                x = self.conv_expand(x)
                x_spatical_score = self.sigmoid(x)
                x = x_origin * x_spatical_score
                x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
                return x

        self.tce = Temporal_Channel_Excitation_conv1d_two_layer(in_channels//2)
        self.sce = Spatial_Channel_Excitation_SE(in_channels//2)

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        spx = torch.split( x, self.width, 1 )
        tce = self.tce(spx[0])
        sce = self.sce(spx[1])
        stce = torch.cat((tce,sce),dim=1)
        return stce

def main():
    x = torch.randn([30, 64, 300, 25], dtype=torch.float32)
    module = MTA_STCE(in_channels=64)
    print(module)
    x = module(x)
    print(x.shape)

if __name__ == "__main__":
    main()