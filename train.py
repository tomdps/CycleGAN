import torch
import torch.nn as nn
from networks import *
from data_loader import *
from utils import *
# import wandb
import pickle

def train(cycle_gan, dataloader1, dataloader2, model_path, run='latest', n_epochs = 100, lambd=10, loss_gan = error(2), loss_cycle = error(1), print_every = 100, save_every=2000, sample_every=500, log=False):

    G1 = cycle_gan.G1
    G2 = cycle_gan.G2
    D1 = cycle_gan.D1
    D2 = cycle_gan.D2
    dl1_iter = iter(dataloader1)
    dl2_iter = iter(dataloader2)
    iter_per_epoch = min(len(dataloader1), len(dataloader2))
    epoch = 1
    buffer_1 = Buffer(50)
    buffer_2 = Buffer(50)
    loss_g = []
    loss_d1 = []
    loss_d2 = []

    for epoch in range(1, n_epochs+1):

        G_loss_acc = 0
        d1_loss_acc = 0
        d2_loss_acc = 0
        
        for iteration in range(iter_per_epoch):
            
            data_1 = dl1_iter.next().cuda()
            data_2 = dl1_iter.next().cuda()

            #inference 
            G1_gen = G1(data_1)
            G2oG1_gen = G2(G1_gen)
            G2_gen = G2(data_2)
            G1oG2_gen = G1(G2_gen)

            #Deactivate gradient computation for discriminators to minimize computational cost for generator training
            for param in D1.parameters():
                param.requires_grad = False
            for param in D2.parameters():
                param.requires_grad = False
            
            #Train generators jointly
            cycle_gan.g_opt.zero_grad()

            G_loss = loss_gan(D1(G1_gen)-1) + lambd*loss_cycle(G2oG1_gen - data_1) + loss_gan(D2(G2_gen)-1) + lambd*loss_cycle(G1oG2_gen - data_2)
            G_loss.backward()

            cycle_gan.g_opt.step()

            #Reactivate gradient computation for discriminators
            for param in D1.parameters():
                param.requires_grad = True
            for param in D2.parameters():
                param.requires_grad = True

            #Train discriminators
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

            G_loss_acc += G_loss
            d1_loss_acc += d1_loss
            d2_loss_acc += d2_loss
        
        print('Epoch [%d/%d], G_loss: %.4f, d1_loss: %.4f, d2_loss: %.4f '
                %(epoch, n_epochs, G_loss_acc.item()/iter_per_epoch, d1_loss_acc.item()/iter_per_epoch, 
                d2_loss_acc.item()/iter_per_epoch))
        
        if epoch % 5 == 0:
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
            
        loss_g.append(G_loss_acc.item()/iter_per_epoch)
        loss_d1.append(G_loss_acc.item()/iter_per_epoch)
        loss_d2.append(G_loss_acc.item()/iter_per_epoch)

        dl1_iter = iter(dataloader1)
        dl2_iter = iter(dataloader2)

    loss = [loss_g, loss_d1, loss_d2]
    loss_filename = os.path.join(model_path, 'loss_%s'%(run))
    with open(loss_filename, 'wb') as handle:
        pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
