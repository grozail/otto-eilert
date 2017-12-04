import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--image_size', type=int, default=32, help='input image size')
parser.add_argument('--nz', type=int, default=100, help='size of the noise')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of descriminator filters')
parser.add_argument('--epoch', type=int, default=10, help='number of train epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--random_seed', type=int, default=666, help='manual seed')

cla = parser.parse_args()


def train():
    import random
    from torchvision import transforms, datasets
    import torch
    import torch.nn as tnn
    import torch.cuda
    from torch.autograd import Variable
    import torch.optim as optim
    import torch.utils.data
    import torchvision.utils as visutils
    # import torch.backends.cudnn as cudnn TODO: add benchmark

    from .model.gan import Descriminator, Generator

    batch_size = int(cla.batch_size)
    image_size = int(cla.image_size)
    learning_rate = float(cla.lr)
    epochs = int(cla.epoch)
    
    random_seed = int(cla.random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cla.cuda:
        torch.cuda.manual_seed(random_seed)

    def custom_loader(path):
        import cv2 as cv
        from PIL import Image
        import numpy as np
        print(path)
        img = cv.imread(path)
        channels = cv.split(img)
        data = channels[0] / 255.0
        return Image.fromarray(data, "F")
    
    cyrilic_dataset = datasets.ImageFolder(root='neural_nets/givemeagan/data/dataset/Cyrillic-small',
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    dataloader = torch.utils.data.DataLoader(cyrilic_dataset, batch_size, shuffle=True, num_workers=2)
    
    n_descriminator_features = int(cla.ndf)
    D = Descriminator(n_descriminator_features)
    
    noize_size = int(cla.nz)
    n_generator_features = int(cla.ngf)
    G = Generator(noize_size, n_generator_features)
    
    inp = torch.FloatTensor(batch_size, 1, image_size, image_size)
    noise = torch.FloatTensor(batch_size, noize_size, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, noize_size, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(batch_size)
    
    real_label = 1
    fake_label = 0
    
    criterion = tnn.BCELoss()
    
    if cla.cuda and torch.cuda.is_available():
        D.cuda()
        G.cuda()
        criterion.cuda()
        inp, label = inp.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    fixed_noise = Variable(fixed_noise)
    
    opt_D = optim.Adam(D.parameters(), learning_rate, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), learning_rate, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            try:
                real_cpu, ids = data
                current_batch_size = real_cpu.size(0)
                if cla.cuda and torch.cuda.is_available():
                    real_cpu = real_cpu.cuda()
        
                    # feed D with real
                D.zero_grad()
                inp.resize_as_(real_cpu).copy_(real_cpu)
                label.resize_((current_batch_size, 1, 1, 1)).fill_(real_label)
                var_inp = Variable(inp)
                var_label = Variable(label)
                output = D(var_inp)
                err_D_real = criterion(output, var_label)
                err_D_real.backward()
                D_x = output.data.mean()
    
                # generate fake image from random noise
                noise.resize_(batch_size, noize_size, 1, 1).normal_(0, 1)
                var_noise = Variable(noise)
                fake = G(var_noise)
    
                # feed D with fake
                var_label = Variable(label.fill_(fake_label))
                output = D(fake.detach())
                err_D_fake = criterion(output, var_label)
                err_D_fake.backward()
                D_G_z1 = output.data.mean()
                err_D = err_D_fake + err_D_real
                opt_D.step()
    
                # update G
                G.zero_grad()
                var_label = Variable(label.fill_(real_label))
                output = D(fake)
                err_G = criterion(output, var_label)
                err_G.backward()
                D_G_z2 = output.data.mean()
                opt_G.step()
    
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, cla.epoch, i, len(dataloader),
                         err_D.data[0], err_G.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    visutils.save_image(real_cpu,
                                        '/opt/ProjectsPy/machine-learning/neural_nets/givemeagan/data/output/real_samples-e{}-b{}.png'.format(epoch, i),
                                        normalize=True)
                    fake = G(fixed_noise)
                    visutils.save_image(fake.data,
                                        '/opt/ProjectsPy/machine-learning/neural_nets/givemeagan/data/output/fake_samples-e{}-b{}.png'.format(epoch, i),
                                        normalize=True)
            except:
                pass
