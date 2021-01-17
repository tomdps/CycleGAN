import torch
import torch.nn as nn
from networks import *
from data_loader import *
from utils import *
import wandb


def train(cycle_gan, dataloader1, dataloader2, model_path, max_iter = 7000, lambd=10, loss_gan = error(2), loss_cycle = error(1), print_every = 100, save_every=2000, sample_every=500, wandb=False):
    G1 = cycle_gan.G1
    G2 = cycle_gan.G2
    D1 = cycle_gan.D1
    D2 = cycle_gan.D2
    dl1_iter = iter(dataloader1)
    dl2_iter = iter(dataloader2)
    iter_per_epoch = min(len(dl1_iter), len(dl2_iter))
        
    for iteration in range(max_iter):
        
        if (iteration +1) % iter_per_epoch == 0:
            dl1_iter = iter(dataloader1)
            dl2_iter = iter(dataloader2)
            
        data_1 = dl1_iter.next().cuda()
        data_2 = dl1_iter.next().cuda()

        #inference
        D1_real = D1(data_2)
        D1_fake = D1(G1(data_1))
        D2_real = D2(data_1)
        D2_fake = D2(G2(data_2))
        G1_gen = G1(data_1)
        D1_gen = D1(G1_gen)
        G2oG1_gen = G2(G1_gen)
        G2_gen = G2(data_2)
        D2_gen = D2(G2_gen)
        G1oG2_gen = G1(G2_gen)

        #Generators training
        for param in D1.parameters():
            param.requires_grad = False
        for param in D2.parameters():
            param.requires_grad = False
        
        cycle_gan.g_opt.zero_grad()

        G_loss = loss_gan(D1_gen-1) + lambd*loss_cycle(G2oG1_gen - data_1) + loss_gan(D1_gen-1) + lambd*loss_cycle(G2oG1_gen - data_1)
        G_loss.backward()

        cycle_gan.g_opt.step()

        for param in D1.parameters():
            param.requires_grad = True
        for param in D2.parameters():
            param.requires_grad = True

        cycle_gan.d_opt.zero_grad()

        d1_loss_real = loss_gan(D1_real-1)
        d1_loss_fake = loss_gan(D1_fake)
        d1_loss = (d1_loss_real + d1_loss_fake)*0.5
        d1_loss.backward()
        
        d2_loss_real = loss_gan(D2_real-1)
        d2_loss_fake = loss_gan(D2_fake)
        d2_loss = (d2_loss_real + d2_loss_fake)*0.5
        d2_loss.backward()
        
        cycle_gan.d_opt.step()
        
        if (iteration+1) % print_every == 0:
            print('Iter [%d/%d], G_loss: %.4f, d1_loss: %.4f, d2_loss: %.4f '
                    %(iteration+1, max_iter, G_loss.item(), d1_loss.item(), 
                    d2_loss.item()))
        
        if (iteration+1) % 5000 == 0:
            g1_path = os.path.join(model_path, 'g1-%d.pkl' %(iteration+1))
            g2_path = os.path.join(model_path, 'g2-%d.pkl' %(iteration+1))
            d1_path = os.path.join(model_path, 'd1-%d.pkl' %(iteration+1))
            d2_path = os.path.join(model_path, 'd2-%d.pkl' %(iteration+1))
            torch.save(G1.state_dict(), g1_path)
            torch.save(G2.state_dict(), g2_path)
            torch.save(D1.state_dict(), d1_path)
            torch.save(D2.state_dict(), d2_path)

        if wandb == True:
            wandb.log({"Generator loss": G_loss , "D1 loss": d1_loss, "d2_loss": d2_loss})


def load_model(model_path, iteration):
    g1_path = os.path.join(model_path, 'g1-%d.pkl' %(iteration))
    g2_path = os.path.join(model_path, 'g2-%d.pkl' %(iteration))
    d1_path = os.path.join(model_path, 'd1-%d.pkl' %(iteration))
    d2_path = os.path.join(model_path, 'd2-%d.pkl' %(iteration))
    model = CycleGAN()
    mode.G1 = torch.load(g1_path)
    mode.G2 = torch.load(g2_path)
    mode.D1 = torch.load(d1_path)
    mode.D2 = torch.load(d2_path)
    return model