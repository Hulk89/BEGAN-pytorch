import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import json
from tensorboardX import SummaryWriter

from image_loader import get_loader
from AE import AutoEncoder, Decoder

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='BEGAN')
    parser.add_argument('--datapath', dest='datapath', required=True,
                        type=str, help='root folder for data')
    parser.add_argument('--batch_size', dest='batch_size', default=16,
                        type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=4,
                        type=int, help='number of workers for data loader')
    parser.add_argument('--config', dest='config', default='began_config.json',
                        type=str, help='config file for BEGAN')
    args = parser.parse_args()


    # dataloader setting
    loader = get_loader(args.datapath, args.batch_size, args.num_workers)

    # load config
    with open(args.config) as f:
        config = json.loads(f.read())

    # print configuration
    print(json.dumps(config, indent=4, sort_keys=True))
    for arg in vars(args):
        print("{}:{}".format(arg, getattr(args, arg)))


    # set Networks
    d = AutoEncoder(channel=config['model']['channels'],
                    n=config['model']['N'],
                    h=config['model']['h'])

    g = Decoder(channel=config['model']['channels'],
                n=config['model']['N'],
                h=config['model']['h'])

    d.cuda()
    g.cuda()
    print(g)
    # hparams 
    lambda_     = config['train']['lambda']
    g_lr = d_lr = config['train']['lr']
    num_iter    = config['train']['iter']
    num_epoch   = config['train']['epoch']
    gamma       = config['train']['gamma']
    beta1       = config['train']['beta1']
    beta2       = config['train']['beta2']

    k = 0.0
    #TODO beta
    d_optimizer = optim.Adam(d.parameters(), lr=d_lr, betas=(beta1, beta2))
    g_optimizer = optim.Adam(g.parameters(), lr=g_lr, betas=(beta1, beta2))

    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, 100, 0.98)
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, 100, 0.98)

    def loss_func(i, o):
        return torch.mean(torch.abs(i-o))
    # tensorboardX
    writer = SummaryWriter()
    try:
        for epoch in range(num_epoch):
            for i, data in enumerate(loader):
                d_scheduler.step()
                g_scheduler.step()

                ### Discriminator Training ###
                d.zero_grad()
                # 1. real image
                real_img = Variable(data[0].cuda())
                restored_real_img = d(real_img)
                # 2. fake image loss
                noise = (torch.rand(args.batch_size, config['model']['h']) - 0.5)*2
                z = Variable(noise).cuda()
                fake_img = g(z).detach()  # Generator에 대해서는 학습하면 안된다. 이미 optimizer에서 학습할 파라미터를 정해주지만, 계산을 빨리 하기위해 이런다.
                restored_fake_img = d(fake_img)

                fake_loss = loss_func(fake_img, restored_fake_img)
                real_loss = loss_func(real_img, restored_real_img)
                # 3. compute discriminator loss
                loss_D = real_loss - k * fake_loss
                loss_D.backward()

                d_optimizer.step()

                ### Generator Training ###
                g.zero_grad()
                
                noise = (torch.rand(args.batch_size, config['model']['h']) - 0.5)*2
                z = Variable(noise).cuda()
                fake_img = g(z)
                restored_fake_img = d(fake_img)  #여기는 detach하면 backpropagation이 안흐를 듯!
                loss_G = loss_func(fake_img, restored_fake_img)
                loss_G.backward()
                
                g_optimizer.step()

                ### compute others ###
                k = k + lambda_*(gamma*real_loss.data[0] - fake_loss.data[0])
                k = max(min(k, 1), 0)
                M_global = real_loss.data[0] + abs(gamma*real_loss.data[0] - fake_loss.data[0])
    
                if i % 1000 == 0:
                    print(M_global, flush=True)
                    step = epoch* 100000000 + i
                    writer.add_scalar('information/M_global', M_global, step)
                    writer.add_scalar('information/k', k, step)
                    writer.add_scalar('information/real_loss', real_loss.data[0], step)
                    writer.add_scalar('information/fake_loss', fake_loss.data[0], step)
                    writer.add_image('RealImg/image', (real_img+1)/2, step)
                    writer.add_image('RealImg/restored', (restored_real_img+1)/2, step) 
                    writer.add_image('FakeImg/image', (fake_img+1)/2, step)
                    writer.add_image('FakeImg/restored', (restored_fake_img+1)/2, step)
                if num_iter != -1 and i == num_iter:
                    raise StopIteration
    except StopIteration:
        pass
    writer.close()

    
