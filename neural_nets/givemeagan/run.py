import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=40, help='input image size')
parser.add_argument('--ns', type=int, default=100, help='size of the noise')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of descriminator filters')
parser.add_argument('--n-epochs', type=int, default=1, help='number of train epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')


p_args = parser.parse_args()
print(p_args)
