import matplotlib.pyplot as plt
import numpy as np
import os
#import itertools
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from train import *



def unnormalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image  

def test(cycle_gan, dataloader, file_to_save = None, dir_num = 1, plot_number = 1, maxiter = np.inf):
    t = tqdm(dataloader, leave=False, total=len(dataloader_photo))
    trans = transforms.ToPILImage()
    plots = 0
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    for i, im in enumerate(t):
        if i== maxiter:
            break
        with torch.no_grad():
            if dir_num ==1:
                gen = cycle_gan.G1(im.to(device)).cpu().detach()
            elif dir_num ==2:
                gen = cycle_gan.G2(im.to(device)).cpu().detach()
            gen = unnormalize(gen)
            img = trans(gen[0]).convert("RGB")
            if file_to_save != None:
                img.save(file_to_save + str(i+1) + ".jpg")

            if plots < plot_number:
                plots +=1
                im = unnormalize(im)
                fig = plt.figure(figsize=(8,8))
                fig.add_subplot(1,2,1)
                plt.title('Original')
                plt.imshow(im[0].permute(1, 2, 0))
                fig.add_subplot(1,2,2)
                plt.title('Generated')
                plt.imshow(img)
    plt.show()
    
    
#if __name__=='main':  
photo_path = ''

data = CycleGanDataset(photo_path, normalize=(True))
    
dataloader = CycleGanDataloader(data)


cycle_gan = CycleGAN()

test(cycle_gan, dataloader, plot_number=3, maxiter = 5)




