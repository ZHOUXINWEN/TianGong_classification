import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from DataLoader import Tiangong
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.pnasnet import pnasnet5large
import argparse

parser = argparse.ArgumentParser(description = 'Tiangong')
parser.add_argument('--dataset_root', default = 'data', type = str)
parser.add_argument('--class_num', default = 6, type = int)
parser.add_argument('--batch_size', default = 16, type = int)
parser.add_argument('--num_workers', default = 4, type = int)
parser.add_argument('--start_iter', default = 0, type = int)
parser.add_argument('--adjust_iter', default = 40000, type = int)
parser.add_argument('--end_iter', default = 60000, type = int)
parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--momentum', default = 0.9, type = float)
parser.add_argument('--weight_decay', default = 5e-4, type = float)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--resume', default = None, type = str)
parser.add_argument('--basenet', default = 'pnasnet', type = str)
#parser.add_argument('--fixblocks', default = 2, type = int)
args = parser.parse_args()

def train():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    # Dataset
    Dataset = Tiangong(root = args.dataset_root, mode = 'trainval')
    Dataloader = data.DataLoader(Dataset, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    # Network
    if args.basenet == 'ResNeXt':
        Network = ResNeXt101_64x4d(args.class_num)    
        net = Network#torch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            Network.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('resnext101_64x4d-e77a0586.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            Network.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(Network.last_linear.weight.data)
            Network.last_linear.bias.data.zero_()
        #for p in Network.features[0].parameters(): p.requires_grad=False
        #for p in Network.features[1].parameters(): p.requires_grad=False
    elif args.basenet == 'pnasnet':
        Network = pnasnet5large(args.class_num,args.resume)    
        net = Network#torch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            Network.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('pnasnet5large-bf079911.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            Network.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(Network.last_linear.weight.data)
            Network.last_linear.bias.data.zero_()
        
    net.train()
    net = net.cuda()

    cl = nn.CrossEntropyLoss()
    # Optimizer
    # Optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum,
                          #weight_decay = args.weight_decay)
    Optimizer = optim.RMSprop(net.parameters(), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay)

    # train
    step = args.start_iter
    loss = 0
    for epoch in range(10000):
        for (imgs, anos) in Dataloader:
            y = net(imgs.cuda())
            Optimizer.zero_grad()
            loss = cl(y, anos.cuda())
            loss.backward()
            Optimizer.step()
            if step % 10 == 0:
                print('step: ' + str(step) + ', loss: ' + repr(loss.data))
            step += 1
            if step == args.adjust_iter:
                adjust_learning_rate(Optimizer, args.gamma)
            if step % 2000 == 0:
                torch.save(Network.state_dict(), 'weights/' + 'Tiangong_RMSProp' + args.basenet + repr(step) + '.pth')
            

def adjust_learning_rate(optimizer, gamma):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()
    
