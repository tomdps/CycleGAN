import matplotlib.pyplot as plt
#import numpy as np
import os
#import itertools
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets
from torchvision import transforms
from PIL import Image


## Dataset
#mini modif 
class CycleGanDataset(Dataset):
    def __init__(self, dir1, dir2, size=(256, 256), normalize=True):
        super().__init__()
        self.dir1 = dir1
        self.dir2 = dir2
        self.dir1_ids = dict()
        self.dir2_ids = dict()
        self.totaldir = dict()
        self.dir1_len = len(os.listdir(self.dir1))
        self.dir2_len = len(os.listdir(self.dir2))
        
        for i, fl in enumerate(os.listdir(self.dir1)):
            self.dir1_ids[i] = fl
            self.totaldir[i] = fl
        for i, fl in enumerate(os.listdir(self.dir2)):
            self.dir2_ids[i] = fl
            self.totaldir[i+self.dir1_len] = fl
            
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])

    def __getitem__(self, image_id):
        if image_id < self.dir1_len:
            path = os.path.join(self.dir1, self.totaldir[image_id])    
        else:
            path = os.path.join(self.dir2, self.totaldir[image_id])
        img = Image.open(path)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.dir1_len + self.dir2_len

    
    def plot_image(self, key):
        plt.figure()
        plt.imshow(self.__getitem__(key)[0])



## Dataloader

class CycleGanDataloader(DataLoader):
    def __init__(self, cycleGANdataset, batch_size=1, pin_memory=True):
        super().__init__(cycleGANdataset, batch_size=1, pin_memory=True)
    
    def unnormalize(self, image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        return image   
    
    def plot_image(self, index, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        images = iter(self)
        img = next(images)
        for i in range(index):
            img = next(images)
        plt.figure()
        plt.imshow(self.unnormalize(img,mean,std)[0].permute(1, 2, 0))

## Test


monet_path = '/Users/alex/Desktop/MVA et Centrale 3A/Deep Learning/Projet/Code/CycleGAN/data/gan-getting-started/monet_jpg'
photo_path = '/Users/alex/Desktop/MVA et Centrale 3A/Deep Learning/Projet/Code/CycleGAN/data/gan-getting-started/photo_jpg'
data = CycleGanDataset(monet_path, photo_path, normalize=(True))


print(data.dir1_len)


data.plot_image(4)

dataloader = CycleGanDataloader(data)
dataloader.plot_image(4)
#dataloader.plot_image(300)


