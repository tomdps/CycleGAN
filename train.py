import torch
import torch.nn as nn
from networks import *
from data_loader import *
from utils import *


def train(cycle_gan, dataloader1, dataloader2, max_iter = 7000, loss_gan = error(2), loss_cycle = error(1)):
    G1 = cycle_gan.G1
    G2 = cycle_gan.G2
    D1 = cycle_gan.D1
    D2 = cycle_gan.D2
    dl1_iter = iter(dataloader1)
    dl2_iter = iter(dataloader2)
    iter_per_epoch = min(len(dl1_iter), len(dl2_iter))
      
    for iteration in max_iter:
      
    	if (iteration +1) % iter_per_epoch == 0:
            dl1_iter = iter(dataloader1)
            dl2_iter = iter(dataloader2)
            
    	data_1 = dl1_iter.next().cuda()
    	data_2 = dl1_iter.next().cuda()
        
        cycle_gan.g_opt.zero_grad()
        cycle_gan.d_opt.zero_grad()
        
        D1_real = D1(data_2)
        d1_loss_real = loss_gan(D1_real-1)
        D2_real = D2(data_1)
        d2_loss_real = loss_gan(D2_real-1)
        d_loss_real = d1_loss_real + d2_loss_real
        d_loss_real.backward()
        cycle_gan.d_opt.step()
        
        D1_fake = D1(G1(data_1))
        d1_loss_fake = loss_gan(D1_fake)
        D2_fake = D2(G2(data_2))
        d2_loss_fake = loss_gan(D2_fake)
        d_loss_fake = d1_loss_fake + d2_loss_fake
        d_loss_fake.backward()
        cycle_gan.d_opt.step()
        
        cycle_gan.g_opt.zero_grad()
        cycle_gan.d_opt.zero_grad()
		
        G1_gen = G1(data_1)
        D1_gen = D1(G1_gen)
        G2oG1_gen = G2(G1_gen)
        G1_loss = loss_gan(D1_gen-1) + loss_cycle(G2oG1_gen - data_1)
        
        d_loss_fake.backward()
        cycle_gan.g_opt.step()
        
        cycle_gan.g_opt.zero_grad()
        cycle_gan.d_opt.zero_grad()

        G2_gen = G2(data_2)
        D2_gen = D2(G2_gen)
        G1oG2_gen = G1(G2_gen)
        G2_loss = loss_gan(D2_gen-1) + loss_cycle(G1oG2_gen - data_2)

        d_loss_fake.backward()
        cycle_gan.g_opt.step()
        
        if (iteration+1) % 100 == 0:
            print('Iter [%d/%d], d_loss_real: %.4f, d_loss_fake: %.4f, G1_loss: %.4f, '
                  'G2_loss: %.4f'
                  %(iteration+1, max_iter, d_loss_real.data[0], d_loss_fake.data[0], 
                    G1_loss.data[0], G2_loss.data[0]))