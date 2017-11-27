import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              3,
                              padding=1)  # for same size
		# convolutional layer init...
        self.conv.weight.data.normal_(0.0, 0.02)
        self.conv.bias.data.fill_(0)

        self.norm = nn.InstanceNorm2d(out_channel)
        self.elu = nn.ELU(inplace=True)
        self.same_channel = in_channel == out_channel

    def forward(self, x):
        y = self.conv(x)
        if self.same_channel:
            y = self.norm(y)
        y = self.elu(y)

        if self.same_channel:
            return y + x
        else:
            return y

class SizeChangeBlock(BasicBlock):
    def __init__(self, in_channel, out_channel, upscale=True):
        super(SizeChangeBlock, self).__init__(in_channel, out_channel)

        if upscale:
            self.sizeChangeBlock = nn.Upsample(scale_factor=2)
        else:
            self.sizeChangeBlock = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        y = super(SizeChangeBlock, self).forward(x) 
        y = self.sizeChangeBlock(y)
        return y


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        channel = kwargs.pop('channel')
        n       = kwargs.pop('n')
        h       = kwargs.pop('h')
        super(Encoder, self).__init__(*args, **kwargs)
        inout_channels = [[channel, n], [n, n], [n, 2*n],
                          [2*n, 2*n], [2*n, 3*n],
                          [3*n, 3*n], [3*n, 4*n],
                          [4*n, 4*n], [4*n, 4*n]]
        subsampling_layer = [2, 4, 6]
        layers = []
        for i, inout_channels in enumerate(inout_channels):
            if i in subsampling_layer:
                layers.append(SizeChangeBlock(inout_channels[0],
                                              inout_channels[1],
                                              upscale=False))
            else:
                layers.append(BasicBlock(inout_channels[0],
                                         inout_channels[1]))
        self.convs = nn.ModuleList(layers)
        self.ff = nn.Linear(8*8*4*n, h)
        # FFN initialize..
        self.ff.weight.data.normal_(0.0, 0.02)
        self.ff.bias.data.fill_(0)

    def forward(self, x):
        res = x
        for conv in self.convs:
            res = conv(res)
        res = res.view(res.size(0), -1)  # flatten
        
        res = self.ff(res)
        return res


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        channel = kwargs.pop('channel')
        n       = kwargs.pop('n')
        h       = kwargs.pop('h')
        super(Decoder, self).__init__(*args, **kwargs)
        inout_channels = [[channel, n], [n, n], [n, 2*n],
                          [2*n, 2*n], [2*n, 3*n],
                          [3*n, 3*n], [3*n, 4*n],
                          [4*n, 4*n], [4*n, 4*n]]
        inout_channels.reverse()
        upsampling_layer = [1, 3, 5]
        self.convs = []
        self.n = n
        #self.act = nn.Tanh()

        layers = []
        for i, inout_channels in enumerate(inout_channels):
            if i in upsampling_layer:
                layers.append(SizeChangeBlock(inout_channels[1],
                                              inout_channels[0],
                                              upscale=True))
            else:
                layers.append(BasicBlock(inout_channels[1],
                                         inout_channels[0]))
        self.convs = nn.ModuleList(layers)
        self.ff = nn.Linear(h, 8*8*4*n)
        self.ff.weight.data.normal_(0.0, 0.02)
        self.ff.bias.data.fill_(0)

    def forward(self, x):
        res = self.ff(x)
        res = res.view(-1, 4*self.n, 8, 8)  # B C H W

        for conv in self.convs:
            res = conv(res)
        #res = self.act(res)  # -1~1까지만...
        return res


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        channel = kwargs.pop("channel")
        n       = kwargs.pop("n")
        h       = kwargs.pop('h')
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.encoder = Encoder(channel=channel, n=n, h=h)
        self.decoder = Decoder(channel=channel, n=n, h=h)

    def forward(self, x):
        return self.decoder(self.encoder(x))

if __name__=='__main__':
    from image_loader import get_loader
    import json
    import argparse
    import torch.optim as optim
#TODO seperate test code into testcase source
    from torch.autograd import Variable
    from tensorboardX import SummaryWriter
    '''
    encoder = Encoder(channel=3, n=64, h=32)
    decoder = Decoder(channel=3, n=64, h=32)
    encoder.cuda()
    decoder.cuda()
    testIn = Variable(torch.randn(10, 3, 64, 64)).cuda()  # B C H W
    print(testIn.size())
    testH = encoder(testIn)
    print(testH.size())
    testOut = decoder(testH)
    print(testOut.size())
    '''
    parser = argparse.ArgumentParser(description='AutoEncoder')
    parser.add_argument('--datapath', dest='datapath', required=True,
                        type=str, help='root folder for data')
    parser.add_argument('--batch_size', dest='batch_size', default=16,
                        type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=4,
                        type=int, help='number of workers for data loader')
    parser.add_argument('--config', dest='config', default='ae_config.json',
                        type=str, help='config file for ae')
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

    gpus = [0]
    # set AutoEncoder
    ae_net = AutoEncoder(channel=config['model']['channels'],
                        n=config['model']['N'],
                        h=config['model']['h'])
    ae_net = ae_net.cuda()
    lambda_   = config['train']['lambda']
    lr        = config['train']['lr']
    num_iter  = config['train']['iter']
    num_epoch = config['train']['epoch']
    #TODO beta
    criterion = nn.L1Loss()
    criterion.cuda()
    optimizer = optim.Adam(ae_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.98)

    # tensorboardX
    writer = SummaryWriter()
    import torchvision.transforms as transforms
    Revert = transforms.Normalize((-1,-1,-1), (2,2,2))
    try:
        for epoch in range(num_epoch):
            for i, data in enumerate(loader):
                scheduler.step()  # in this case, we decay lr with some steps of iteration
                optimizer.zero_grad()

                input_img = torch.autograd.Variable(data[0].cuda())
                output_img = ae_net(input_img)
                #loss = criterion(input_img, output_img.detach())
                loss = torch.mean(torch.abs(input_img-output_img))
                loss.backward()
                optimizer.step()
            
                if i % 1000 == 0:
                    print(loss.data[0], flush=True)
                    writer.add_image('Input image', (input_img+1)/2, epoch* 10000000 + i)
                    writer.add_image('output image', (output_img+1)/2, epoch* 10000000 + i)
                if num_iter != -1 and i == num_iter:
                    raise StopIteration
    except StopIteration:
        pass
    writer.close()
