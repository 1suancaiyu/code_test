from .action_net import Action



inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))

# image input
"""
x = torch.zeros([16,30,192,10,10], dtype=torch.float32)
n, t, c, h, w = x.size()
x = x.view(n * t, c, h, w)
act = Action(inception_3a_1x1)
act(x)
"""

# skeleton input
x = torch.zeros([16,30,192,10,10], dtype=torch.float32)