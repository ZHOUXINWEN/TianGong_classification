import torch
import torch.utils.data as data
import cv2
import numpy as np
import csv
import os

CLASSES = ('DESERT', 'MOUNTAIN', 'OCEAN', 'FARMLAND', 'LAKE', 'CITY')

class Tiangong(data.Dataset):
    def __init__(self, root, mode):
        # mode can be valued as "train", "val", "trainval"
        self.root = root
        with open(os.path.join(self.root, mode + '.csv')) as f:
            reader = csv.reader(f)
            self.ids = [row for row in reader]
        self.cls_to_id = dict(zip(CLASSES, range(len(CLASSES))))
        self.mean = np.array([113.4757, 112.0985, 102.8271])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.ids[index]
        Img = cv2.imread(os.path.join(self.root, 'train', item[0]))
        Img = cv2.resize(Img, (224, 224)) #224 for ResNeXt, 331 for PNASNet5Large
        Img = Img.astype(np.float)
        Img = Img[:, :, (2, 1, 0)] - self.mean
        
        Anno = self.cls_to_id[item[1]]
        return torch.from_numpy(Img).permute(2, 0, 1).type(torch.FloatTensor), Anno

class TiangongResult(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = os.listdir(os.path.join(self.root, 'test'))
        self.mean = np.array([113.4757, 112.0985, 102.8271])

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        item = self.ids[index]
        Img = cv2.imread(os.path.join(self.root, 'test', item))
        Img = cv2.resize(Img, (224, 224))
        Img = Img.astype(np.float)
        Img = Img[:, :, (2, 1, 0)] - self.mean

        return torch.from_numpy(Img).permute(2, 0, 1).type(torch.FloatTensor), item


        
        
