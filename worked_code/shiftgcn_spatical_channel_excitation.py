"""
from 87.8% to 88.14% (+0.34)
spatial channel interaction
give spatial info to channel
"""

# Spatial_Channel_Excitation
class Spatial_Channel_Excitation(nn.Module):
    def __init__(self):
        super(Spatial_Channel_Excitation, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  Spatial_Channel_Excitation")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        x = x.mean(1, keepdim=True)
        x = self.conv(x)
        x_spatical_score = self.sigmoid(x)
        x = x_origin * x_spatical_score
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x