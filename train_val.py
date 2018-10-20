import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
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
parser.add_argument('--basenet', default = 'ResNeXt', type = str)
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
#parser.add_argument('--fixblocks', default = 2, type = int)
args = parser.parse_args()

def main():
    #create model
    best_prec1 = 0
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    if args.basenet == 'ResNeXt':
        model = ResNeXt101_64x4d(args.class_num)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            Network.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('resnext101_64x4d-e77a0586.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            model.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(model.last_linear.weight.data)
            model.last_linear.bias.data.zero_()
        for p in model.features[0].parameters(): p.requires_grad=False
        for p in model.features[1].parameters(): p.requires_grad=False
    elif args.basenet == 'pnasnet':
        model = pnasnet5large(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('pnasnet5large-bf079911.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            model.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(model.last_linear.weight.data)
            model.last_linear.bias.data.zero_()
    model = model.cuda()
    cudnn.benchmark = True

    # Dataset
    Dataset_train = Tiangong(root = args.dataset_root, mode = 'trainval')
    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    Dataset_val = Tiangong(root = args.dataset_root, mode = 'val')
    Dataloader_val = data.DataLoader(Dataset_val, batch_size = 1,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    criterion = nn.CrossEntropyLoss().cuda()

    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(Optimizer, epoch)

        # train for one epoch
        train(Dataloader_train, model, criterion, Optimizer, epoch)    #train(Dataloader_train, Network, criterion, Optimizer, epoch)

        # evaluate on validation set
        #prec1 = validate(Dataloader_val, model, criterion)  #prec1 = validate(Dataloader_val, Network, criterion)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        #if is_best:
        torch.save(model.state_dict(), 'weights/fixblock_Newtrain_'+ args.basenet +'/'+ '_Tiangong_RMSProp_' + repr(epoch) + '.pth')

        

def train(Dataloader,model, criterion, optimizer, epoch):
    # Priors

    # Dataset
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    model.train()
    model = model.cuda()

    #cl = nn.CrossEntropyLoss()
    # Optimizer

    #Optimizer = optim.RMSprop(net.parameters(), lr = args.lr, momentum = args.momentum,
                          #weight_decay = args.weight_decay)

    # train
    end = time.time()
    for i, (input,anos) in enumerate(Dataloader):
        data_time.update(time.time() - end)

        target = anos.cuda(async=True)

        with torch.no_grad():
            input_var = Variable(input.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_var.cuda())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(Dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))        

'''    for epoch in range(10000):
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
                torch.save(Network.state_dict(), 'weights/' + 'Tiangong_SGD' + args.basenet + repr(step) + '.pth')
'''            
def validate(val_loader,model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = Variable(input.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i+1, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.97 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
    
