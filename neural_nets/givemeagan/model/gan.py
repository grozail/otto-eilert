import torch
import torch.nn as tnn

import torch.optim as optim
import torch.utils.data
import torchvision.datasets
from torch.autograd import Variable


# ===================== GENERATOR =========================
class Generator(tnn.Module):
    def __init__(self, noise_size, n_classes, n_features):
        super(Generator, self).__init__()
        self.main_net = tnn.Sequential(
            tnn.ConvTranspose2d(noise_size + n_classes, n_features * 4, 4, 1, 0, bias=False),
            tnn.BatchNorm2d(n_features * 4),
            tnn.ReLU(True),
            # n_features* 4 x 4 x 4
            tnn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1, bias=False),
            tnn.BatchNorm2d(n_features * 4),
            tnn.ReLU(True),
            # n_features * 2 x 8 x 8
            tnn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
            tnn.BatchNorm2d(n_features * 2),
            tnn.ReLU(True),
            # n_features x 16 x 16
            tnn.ConvTranspose2d(n_features, 1, 4, 2, 1, bias=False),
            # 1 x 32 x 32
            tnn.Tanh(),
        )
        
    def forward(self, inp):
        return self.main_net(inp)


class Descriminator(tnn.Module):
    def __init__(self, n_features):
        super(Descriminator, self).__init__()
        self.main_net = tnn.Sequential(
            # input signals 1 x 32 x 32
            tnn.Conv2d(1, n_features, 4, 2, 1, bias=False),
            tnn.LeakyReLU(0.1, True),
            # n_features x 16 x 16
            tnn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            tnn.BatchNorm2d(n_features * 2),
            tnn.LeakyReLU(0.1, True),
            # n_features * 2 x 8 x 8
            tnn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            tnn.BatchNorm2d(n_features * 4),
            tnn.LeakyReLU(0.1, True),
            # n_features * 4 x 4 x 4
            tnn.Conv2d(n_features * 4, 1, 4, 1, 0, bias=False),
            tnn.Sigmoid(),
        )
    
    def forward(self, inp):
        return self.main_net(inp)