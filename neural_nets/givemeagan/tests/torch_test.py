import torch
import torch.nn as tnn
D = tnn.Sequential(
    tnn.ConvTranspose2d(100, 64, 4, 1, 0, bias=False),
    tnn.BatchNorm2d(64),
    tnn.ReLU(True)
)

if __name__ == '__main__':
    for child in D.children():
        if isinstance(child, tnn.ConvTranspose2d):
            print(child.out_channels)
