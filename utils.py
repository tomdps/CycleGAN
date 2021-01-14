import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from networks import *
from data_loader import *
from utils import *

def error(n):
    def loss_n(tensor):
    	return torch.mean(tensor**n)
	return loss_n

def init_model(cycle_gan, optimizer = None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, init_std=0.01):
    # weights initialization
    #N(0,0.01)
    def init_weights(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal(m.weight.data, 0.0, init_std)
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
    g_params = list(cycle_gan.G1.parameters()) + list(cycle_gan.G2.parameters())
    d_params = list(cycle_gan.D1.parameters()) + list(cycle_gan.D2.parameters())

    g_optimizer = Adam(g_params, lr, betas, eps, weight_decay, amsgrad)
    d_optimizer = Adam(d_params, lr, betas, eps, weight_decay, amsgrad)
    return g_optimizer, d_optimizer