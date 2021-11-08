import torch
import torch.nn as nn
from motion_excitation import *
from ConvTemporalGraphical import ConvTemporalGraphical
from MTA import Res2Net
from Bottle2neck import Bottle2neck
# me block
'''
x = torch.zeros([16,300,32,10,10], dtype=torch.float32)
me = MEModule(channel=32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
tmp = me(x)
'''

# MTA block

model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)


# st_gcn gcn
'''
A = torch.zeros([3,25,25], dtype=torch.float32) #
x = torch.zeros([16,3,300,25], dtype=torch.float32) # n,c,t,v
gcn = ConvTemporalGraphical(in_channels=3, out_channels=2, kernel_size=3)
'''

