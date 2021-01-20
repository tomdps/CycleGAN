import torch
import torch.nn as nn
from networks import *
from data_loader import *
from utils import *
import wandb


def train(cycle_gan, dataloader1, dataloader2, model_path, run='latest', max_iter = 7000, lambd=10, loss_gan = error(2), loss_cycle = error(1), print_every = 100, save_every=2000, sample_every=500, log=False):

    run = wandb.init(project="cyclegan")
    G1 = cycle_gan.G1
    G2 = cycle_gan.G2
    D1 = cycle_gan.D1
    D2 = cycle_gan.D2
    dl1_iter = iter(dataloader1)
    dl2_iter = iter(dataloader2)
    iter_per_epoch = min(len(dl1_iter), len(dl2_iter))
    epoch = 1
    buffer_1 = Buffer(50)
    buffer_2 = Buffer(50)
        
    for iteration in range(max_iter):
        
        if (iteration +1) % iter_per_epoch == 0:
            dl1_iter = iter(dataloader1)
            dl2_iter = iter(dataloader2)
            
        data_1 = dl1_iter.next().cuda()
        data_2 = dl1_iter.next().cuda()

        #inference 
        G1_gen = G1(data_1)
        G2oG1_gen = G2(G1_gen)
        G2_gen = G2(data_2)
        G1oG2_gen = G1(G2_gen)

        #Generators training
        for param in D1.parameters():
            param.requires_grad = False
        for param in D2.parameters():
            param.requires_grad = False
        
        cycle_gan.g_opt.zero_grad()

        G_loss = loss_gan(D1(G1_gen)-1) + lambd*loss_cycle(G2oG1_gen - data_1) + loss_gan(D2(G2_gen)-1) + lambd*loss_cycle(G1oG2_gen - data_2)
        G_loss.backward()

        cycle_gan.g_opt.step()

        #Discriminators training
        for param in D1.parameters():
            param.requires_grad = True
        for param in D2.parameters():
            param.requires_grad = True

        cycle_gan.d_opt.zero_grad()


        gen_1 = buffer_1.sample(G1_gen)
        d1_loss_real = loss_gan(D1(data_2)-1)
        d1_loss_fake = loss_gan(D1(gen_1.detach()))
        d1_loss = 0.5*(d1_loss_real + d1_loss_fake)
        d1_loss.backward()
        
        gen_2 = buffer_2.sample(G2_gen)
        d2_loss_real = loss_gan(D2(data_1)-1)
        d2_loss_fake = loss_gan(D2(gen_2.detach()))
        d2_loss = 0.5*(d2_loss_real + d2_loss_fake)
        d2_loss.backward()
        
        cycle_gan.d_opt.step()
        
        if (iteration+1) % iter_per_epoch == 0:
            print('Epoch [%d/%d], G_loss: %.4f, d1_loss: %.4f, d2_loss: %.4f '
                    %(epoch, max_iter, G_loss.item(), d1_loss.item(), 
                    d2_loss.item()))
        
        if (iteration+1) % iter_per_epoch*5 == 0:
            g1_path = os.path.join(model_path, 'g1-%s-%d.pkl' %(run,epoch))
            g2_path = os.path.join(model_path, 'g2-%s-%d.pkl' %(run,epoch))
            d1_path = os.path.join(model_path, 'd1-%s-%d.pkl' %(run,epoch))
            d2_path = os.path.join(model_path, 'd2-%s-%d.pkl' %(run,epoch))
            torch.save(G1.state_dict(), g1_path)
            torch.save(G2.state_dict(), g2_path)
            torch.save(D1.state_dict(), d1_path)
            torch.save(D2.state_dict(), d2_path)

        if log == True:
            run.log({"Generator loss": G_loss , "D1 loss": d1_loss, "d2_loss": d2_loss})
        
        epoch +=1


def load_model(model_path, version):
    g1_path = os.path.join(model_path, 'g1-%s.pkl' %(version))
    g2_path = os.path.join(model_path, 'g2-%s.pkl' %(version))
    d1_path = os.path.join(model_path, 'd1-%s.pkl' %(version))
    d2_path = os.path.join(model_path, 'd2-%s.pkl' %(version))
    model = CycleGAN()
    init_model(model)
    model.G1.load_state_dict(torch.load(g1_path))
    model.G2.load_state_dict(torch.load(g2_path))
    model.D1.load_state_dict(torch.load(d1_path))
    model.D2.load_state_dict(torch.load(d2_path))
    return model