import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import random
import itertools

from networks import *
from data_loader import *
from utils import *

def error(n):
    def loss_n(tensor):
        return torch.mean(torch.abs(tensor**n))
    return loss_n

def init_model(cycle_gan, init = 'normal', optimizer = None, lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, init_std=0.02):

    init_type = init
    def init_weights(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_std)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, init_std)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm2d') != -1: 
            nn.init.normal_(m.weight.data, 1.0, init_std)
            nn.init.constant_(m.bias.data, 0.0)

    cycle_gan.G1.apply(init_weights)
    cycle_gan.G2.apply(init_weights)
    cycle_gan.D1.apply(init_weights)
    cycle_gan.D2.apply(init_weights)

    #optimizer
    if optimizer == None:
        g_opt, d_opt = init_optimizer(cycle_gan, lr, betas, eps, weight_decay, amsgrad)
    else:
        g_opt, d_opt = optimizer
    if torch.cuda.is_available():
        cycle_gan.G1.cuda()
        cycle_gan.G2.cuda()
        cycle_gan.D1.cuda()
        cycle_gan.D2.cuda()
                                      
    cycle_gan.g_opt = g_opt
    cycle_gan.d_opt = d_opt
                                      
                                      

def init_optimizer(cycle_gan, lr, betas, eps, weight_decay, amsgrad):
    g_params = itertools.chain(cycle_gan.G1.parameters(), cycle_gan.G2.parameters())
    d_params = itertools.chain(cycle_gan.D1.parameters(), cycle_gan.D2.parameters())

    g_optimizer = Adam(g_params, lr, betas, eps, weight_decay, amsgrad)
    d_optimizer = Adam(d_params, lr, betas, eps, weight_decay, amsgrad)
    return g_optimizer, d_optimizer

class Buffer():
    def __init__(self, size):
        self.size = size
        self.images = []
        self.n_images = 0

    def sample(self, images):
        sample = []
        for image in images:
            if self.n_images < self.size:
                self.images.append(image)
                self.n_images += 1
                sample.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    pick_image = random.randint(0, self.size - 1)
                    sampled_image = self.images[pick_image].clone()
                    self.images[pick_image] = image
                    sample.append(sampled_image)
                else:
                    sample.append(image)
        sample = torch.cat(sample, 0)
        return sample

def normalization_layer(layer_type):
    if layer_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine = True, track_running_stats = True)
    elif layer_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine = False, track_running_stats = False)
    else:
        print('%s : Norm layer not implemented'%(layer_type))