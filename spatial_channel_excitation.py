import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import random

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

def main():
    x = torch.randn([30, 64, 300, 25], dtype=torch.float32)
    mtm = Spatial_Channel_Excitation()
    x = mtm(x)
    print(x.shape)

if __name__ == "__main__":
    main()