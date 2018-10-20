import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from DataLoader import TiangongResult
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.pnasnet import pnasnet5large
import os

CLASSES = ('DESERT', 'MOUNTAIN', 'OCEAN', 'FARMLAND', 'LAKE', 'CITY')

def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset = TiangongResult(root = 'data')
    Dataloader = data.DataLoader(Dataset, 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)

    # Network
    
    #Network = pnasnet5large(6, None)
    Network = ResNeXt101_64x4d(6)
    net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True
    
    Network.load_state_dict(torch.load('weights/Newtrain_ResNeXt/_Tiangong_RMSProp_21.pth'))
    net.eval()

    filename = 'Newtrain_ResNeXt_Tiangong_RMSProp_21.csv'
    # Result file preparation
    if os.path.exists(filename):
        os.remove(filename)
    os.mknod(filename)

    f = open(filename, 'w')

    for (imgs, anos) in Dataloader:
        imgs = imgs.cuda()
        preds = net.forward(imgs)
        _, pred = preds.data.topk(1, 1, True, True)
        f.write(anos[0] + ',' + CLASSES[pred[0][0]] + '\r\n')

if __name__ == '__main__':
    GeResult()
        
