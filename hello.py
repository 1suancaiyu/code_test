import torch
import torch.nn as nn
from motion_excitation import *

# me block
'''
x = torch.zeros([16,300,32,10,10], dtype=torch.float32)
me = MEModule(channel=32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
tmp = me(x)
'''

# MTA block


# tmp
for i in range(0,3):
    print(i)