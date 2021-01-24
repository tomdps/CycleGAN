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
    def __init__(self, dir,  size=(256, 256), normalize=True):
        super().__init__()
        self.dir = dir
        self.dir_ids = dict()

         
        for i, fl in enumerate(os.listdir(self.dir)):
            self.dir_ids[i] = fl
            
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
        path = os.path.join(self.dir, self.dir_ids[image_id])    
        img = Image.open(path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dir_ids.keys())

    
    def plot_image(self, key):
        plt.figure()
        plt.imshow(self.__getitem__(key)[0])



## Dataloader

class CycleGanDataloader(DataLoader):
    def __init__(self, cycleGANdataset, batch_size=1, pin_memory=True):
        super().__init__(cycleGANdataset, batch_size=batch_size, pin_memory=True)
    
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

if __name__=='main':
    monet_path = ''
    photo_path = ''
    data1 = CycleGanDataset(monet_path, normalize=(True))
    data2 = CycleGanDataset(photo_path, normalize=(True))
    
    print(data1.__len__())
    print(data2.__len__())
    
    data1.plot_image(4)
    data2.plot_image(4)
    
    dataloader1 = CycleGanDataloader(data1)
    dataloader1.plot_image(4)
    
    dataloader2 = CycleGanDataloader(data2)
    dataloader2.plot_image(4)

