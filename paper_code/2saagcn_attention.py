"""
2sagcn attention
"""

import torch.nn as nn
class Spatial_Attention:
    def __init__(self, out_channels, num_jpts):
        super(Spatial_Attention, self).__init__()
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)

    def forward(self, x):
        # spatial attention
        x_origin = x
        x = x.mean(-2)  # N C V
        x = self.sigmoid(self.conv_sa(x))
        x = x_origin * x.unsqueeze(-2) + x
        return x