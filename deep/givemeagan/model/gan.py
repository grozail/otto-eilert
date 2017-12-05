import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--image_size', type=int, default=32, help='input image size')
parser.add_argument('--nz', type=int, default=100, help='size of the noise')
parser.add_argument('--ngf', type=int, default=64, help='number of generator features')
parser.add_argument('--ndf', type=int, default=64, help='number of descriminator features')
parser.add_argument('--epoch', type=int, default=10, help='number of train epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.005')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--random_seed', type=int, default=666, help='manual seed')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, datasets
import torch.cuda
from torch.autograd import Variable
import torchvision.utils as visutils

import random

CUDA = torch.cuda.is_available()
print('CUDA STATE: ', CUDA)

real_label = 1
fake_label = 0
criterion = nn.BCELoss()

cyrilic_dataset = datasets.ImageFolder(root='deep/givemeagan/data/dataset/Cyrillic-small',
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
dataloader = torch.utils.data.DataLoader(cyrilic_dataset, int(args.batch_size), shuffle=True, num_workers=1)
# ===================== GENERATOR =========================
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.noise_size = int(args.nz)
        noise_size = self.noise_size
        self.fixed_noise = torch.FloatTensor(int(args.batch_size), noise_size, 1, 1).normal_(0, 1)
        n_features = int(args.ngf)
        self.GENERATOR = nn.Sequential(
            nn.ConvTranspose2d(noise_size, n_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(True),
            # n_features* 4 x 4 x 4
            nn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(True),
            # n_features * 2 x 8 x 8
            nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(True),
            # n_features x 16 x 16
            nn.ConvTranspose2d(n_features, 3, 4, 2, 1, bias=False),
            # 3 x 32 x 32
            nn.Tanh(),
        )
        self.GENERATOR.apply(weights_init)
        if CUDA:
            self.GENERATOR.cuda()
            self.fixed_noise = Variable(self.fixed_noise.cuda())
        self.optimizer = optim.Adam(self.GENERATOR.parameters(), float(args.lr), betas=(0.9, 0.999))
        self.GENERATOR.train()
        
    def forward(self, inp):
        return self.GENERATOR(inp)
    
    def train_iteration(self, real_input, descriminator):
        current_batch_size = real_input.size(0)
        noise = torch.FloatTensor(current_batch_size, self.noise_size, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(current_batch_size, 1, 1, 1).fill_(real_label)
        if CUDA:
            noise, label = noise.cuda(), label.cuda()
        var_noise, var_label = Variable(noise), Variable(label)
        self.optimizer.zero_grad()
        fake_sample = self.GENERATOR(var_noise)
        output = descriminator(fake_sample)
        generation_error = criterion(output, var_label)
        generation_error.backward()
        probability_generated_is_real = output.data.mean()
        self.optimizer.step()
        return fake_sample, probability_generated_is_real, generation_error
    
    def gen_fixed(self):
        return self.GENERATOR(self.fixed_noise)


# ===================== DESCRIMINATOR =========================
class Descriminator(nn.Module):
    
    def __init__(self):
        super(Descriminator, self).__init__()
        n_features = int(args.ngf)
        self.DESCRIMINATOR = nn.Sequential(
            # input signals 1 x 32 x 32
            nn.Conv2d(3, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            # n_features x 16 x 16
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.1, True),
            # n_features * 2 x 8 x 8
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.1, True),
            # n_features * 4 x 4 x 4
            nn.Conv2d(n_features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.DESCRIMINATOR.apply(weights_init)
        if CUDA:
            self.DESCRIMINATOR.cuda()
        self.optimizer = optim.Adam(self.DESCRIMINATOR.parameters(), float(args.lr), betas=(0.9, 0.999))
        self.DESCRIMINATOR.train()
    
    def forward(self, inp):
        return self.DESCRIMINATOR(inp)
    
    # out parameter for future use
    def train_iteration(self, real_input, fake_input, out):
        current_batch_size = real_input.size(0)
        label = torch.FloatTensor(current_batch_size, 1, 1, 1).fill_(real_label)
        if CUDA:
            real_input, label = real_input.cuda(), label.cuda()
        var_inp, var_label = Variable(real_input), Variable(label),
        self.optimizer.zero_grad()
        output = self.DESCRIMINATOR(var_inp)
        error_on_real = criterion(output, var_label)
        error_on_real.backward()
        probability_real_is_real = output.data.mean()
        
        var_label = Variable(label.fill_(fake_label))
        output = self.DESCRIMINATOR(fake_input.detach())
        error_on_fake = criterion(output, var_label)
        error_on_fake.backward()
        probability_fake_is_real = output.data.mean()
        full_error = error_on_fake + error_on_real
        self.optimizer.step()
        
        return probability_real_is_real, probability_fake_is_real, full_error


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class HaakonGAN:
    def __init__(self):
        random_seed = int(args.random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if CUDA:
            torch.cuda.manual_seed(random_seed)
        self.G = Generator()
        self.D = Descriminator()
        if CUDA:
            self.G.cuda()
            self.D.cuda()
    
    def train(self):
        for epoch in range(1, int(args.epoch)):
            for i, (x, label) in enumerate(dataloader):
                sample, probability_generated_is_real, generation_error = self.G.train_iteration(x, self.D)
                probability_real_is_real, p_fake_is_real, descrimination_error = self.D.train_iteration(x, sample, None)
                print('[{}/{}][{}/{}] | Loss_D: {:.4f} | Loss_G: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f} / {:.4f}'
                      .format(epoch, int(args.epoch), i, len(dataloader),
                              descrimination_error.data[0], generation_error.data[0],
                              probability_real_is_real, p_fake_is_real, probability_generated_is_real))
                if i % 100 == 0:
                    visutils.save_image(x,
                                        'deep/givemeagan/data/output/real_samples-e{}-b{}.png'.format(epoch, i),
                                        normalize=True)
                    fake = self.G.gen_fixed()
                    visutils.save_image(fake.data,
                                        'deep/givemeagan/data/output/fake_samples-e{}-b{}.png'.format(epoch, i),
                                        normalize=True)
