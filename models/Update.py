#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics
import torchvision.utils as vutils
from itertools import cycle





class DatasetSplit_labelid(Dataset):
    def __init__(self, dataset, idxs, mask_vector):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.mv = mask_vector
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.dataset[self.idxs[item]][0]
        label = int(self.mv[self.idxs[item]][2])
        l_or_un = self.mv[self.idxs[item]][0]
        return image, (label, l_or_un)

class DatasetSplit_mask(Dataset):
    def __init__(self, dataset, idxs, mask_vector):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.mv = mask_vector
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = int(self.mv[self.idxs[item]][2])

        return image, label

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def distrib2odds(Dx):
    odds = Dx/(1-Dx)
    return odds

def odds2distrib(odds):
    Dx = odds/(1+odds)
    return Dx


class LocalUpdate(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.selected_clients = []

        self.ldr_label = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs_label, shuffle=True,drop_last=True)
#         self.ldr_label = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs_label, shuffle=True)
        self.ldr_unlabel = DataLoader(DatasetSplit_mask(dataset, idxs2, mask_vector=maskv),batch_size=self.args.local_bs_unlabel, shuffle=True,drop_last=True)

    def GAN_train(self, C, G, D, args, img_size, sampled_time,data_ratio, wd_c = None,wd_d = None,wd_g = None):
        G.train()
        D.train()
        C.train()
        # train and update
        opt_lr = 1e-3
        if sampled_time % 10 == 0:
            opt_lr = opt_lr * 0.5
        
        optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerC = torch.optim.Adam(C.parameters(), lr=opt_lr, betas=(0.9, 0.999), eps=1e-08)
        optimizerC_pre = torch.optim.SGD(filter(lambda p: p.requires_grad, C.parameters()), lr=0.01, momentum=0.5)

        # Loss functions
        BCE_loss = torch.nn.BCELoss().cuda()
        BCE_loss1 = torch.nn.BCELoss(reduce = False).cuda()
        CE_loss = self.loss_func
        
        # label preprocess
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1


        G_epoch_loss = []
        D_epoch_loss = []
        C_epoch_loss = []
        
        alpha_P = 0.5
        alpha_pseudo = 0.1
        alpha_threshold = 250
        beta = 1.0

        for epoch in range(self.args.local_ep):
            D_losses = []
            G_losses = []
            C_losses = []
            C_preloss = []
            
#             if sampled_time * 5 < 200:
#                 for batch_idx, (img, label) in enumerate(self.ldr_label):
#                      # print(batch_idx)
#                     img, label = img.to(self.args.device), label.to(self.args.device).long()
#                     C.zero_grad()
#                     log_probs = C(img)
#                     C_loss = self.loss_func(log_probs, label)
#                     C_loss.backward()
#                     optimizerC_pre.step()
                    
            
            
            
            
            for batch_idx, ((image1, labels1),(image2,labels2)) in enumerate(zip(cycle(self.ldr_label),self.ldr_unlabel)):
#             for batch_idx, ((image1, labels1),(image2,labels2)) in enumerate(zip(self.ldr_label,self.ldr_unlabel)):
                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()
                image_u, label_u = image2.to(self.args.device), labels2.to(self.args.device).long()
                label_ld = fill[label_l].to(self.args.device)
                label_ud = fill[label_u].to(self.args.device)
                label_lg = onehot[label_l].to(self.args.device)
                label_ug = onehot[label_u].to(self.args.device)

                mini_batch_l = image_l.size()[0]
                mini_batch_u = image_u.size()[0]

                # Adversarial ground truths
                y_real_l = torch.ones(mini_batch_l).float()
                y_fake_l = torch.zeros(mini_batch_l).float()
#                 y_fake_l = torch.ones(mini_batch_l).float()
                
                y_real_l, y_fake_l = y_real_l * 0.9, y_fake_l * 0.1
                y_real_l, y_fake_l = y_real_l.to(self.args.device).float(), y_fake_l.to(self.args.device).float()

                z_l = torch.randn((mini_batch_l, 100)).view(-1, 100, 1, 1).to(self.args.device)
#                 z_l = z_l.long()

                # Adversarial ground truths
                y_real_u = torch.ones(mini_batch_u).float()
                y_fake_u = torch.zeros(mini_batch_u).float()
                # label reverse
#                 y_fake_u = torch.ones(mini_batch_u).float()
#                 y_real_u = torch.ones(mini_batch_u).float()
#                 y_real_u, y_fake_u = y_real_u * 0.9, y_fake_u * 0.1
                y_real_u, y_fake_u = y_real_u.to(self.args.device).float(), y_fake_u.to(self.args.device).float()

                z_u = torch.randn((mini_batch_u, 100)).view(-1, 100, 1, 1).to(self.args.device)
#                 z_u = z_u.long()

                # A Game with Three Players

                ######################
                # train Discriminator by labeled data
                ######################
                D.zero_grad()
                C.zero_grad()
                G.zero_grad()
                
                log_probsD_real = D(image_l, label_ld)
                D_loss_real = torch.mean(BCE_loss(log_probsD_real, y_real_l)) # Mean(log(D(x,y)))


                ######################
                # train Generator by labeled data
                ######################
                
                img_g = G(z_l, label_lg)
                log_probsD_g = D(img_g, label_ld)
                odds = distrib2odds(log_probsD_g)
                Dx = odds2distrib(odds*data_ratio)


                D_loss_fake = torch.mean(BCE_loss(log_probsD_g, y_fake_l)) # (1-alpha)*Mean(log(1-D(x_g,y)))


                ######################
                # train Classifer C by labeled data
                ######################
                
                log_probsC = C(image_l)
                C_real_loss = self.loss_func(log_probsC, label_l) 
                Rl = torch.mean(C_real_loss) # RL


                ######################
                # utilizing of unlabeled data
                ######################
                pseudo_label = C(image_u)
#                 print(pseudo_label)
                max_c = torch.argmax(pseudo_label).float()
                p_c = F.softmax(pseudo_label).float()
                _ = torch.argmax(pseudo_label, dim=1).long()
#                 print(_)
                pseudo_labeld = fill[_].to(self.args.device)
                log_probsD_fake = D(image_u, pseudo_labeld) 

                D_loss_cla = torch.mean(BCE_loss(log_probsD_fake, y_fake_u))# alpha*Mean(log(1-D(x,y_c)))

#                 C_loss_dis = torch.mean(max_c * self.loss_func(log_probsD_fake, y_real_u))
                C_loss_dis = torch.mean(p_c*BCE_loss1(log_probsD_fake, y_real_u))

                log_probsC = C(img_g)
#                 pseudo_label_g = torch.argmax(log_probsC, dim=1)
                Rp = torch.mean(self.loss_func(log_probsC, label_l)) # Rp

                if wd_d == None:
                    D_loss = D_loss_real + (1 - alpha_P) * D_loss_fake + alpha_P * D_loss_cla
                else:
                    D_loss = D_loss_real + (1 - alpha_P) * D_loss_fake + alpha_P * D_loss_cla + beta * wd_d
                
                if wd_g == None:
                    G_loss = (1 - alpha_P) * torch.mean(BCE_loss(Dx, y_real_l))
                else:
                    G_loss = (1 - alpha_P) * torch.mean(BCE_loss(Dx, y_real_l)) + beta * wd_g
#                 G_loss = (1 - alpha_P) * torch.mean(BCE_loss(log_probsD_g, y_fake_l))
                
                if wd_c == None:
                    if sampled_time * self.args.local_ep > alpha_threshold:
                        C_loss = 0.01 * alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp
#                     C_loss = alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp
                        C_loss.backward(retain_graph=True)
                        optimizerC.step()
                    elif sampled_time * self.args.local_ep <=alpha_threshold and sampled_time * self.args.local_ep >= 200:
#                     C_loss =  0.01 * alpha_P * C_loss_dis + Rl 
                        C_loss = alpha_P * C_loss_dis + Rl
                        C_loss.backward(retain_graph=True)
                        optimizerC.step()
                    else: 
                        C_loss = Rl
                        C_loss.backward(retain_graph=True)
                        optimizerC_pre.step()
                else:
                    if sampled_time * self.args.local_ep > alpha_threshold:
                        C_loss = 0.01 * alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp + beta * wd
#                     C_loss = alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp
                        C_loss.backward(retain_graph=True)
                        optimizerC.step()
                    elif sampled_time * self.args.local_ep <=alpha_threshold and sampled_time * self.args.local_ep >= 200:
#                     C_loss =  0.01 * alpha_P * C_loss_dis + Rl 
                        C_loss = alpha_P * C_loss_dis + Rl+beta * wd
                        C_loss.backward(retain_graph=True)
                        optimizerC.step()
                    else: 
                        C_loss = Rl+beta * wd
                        C_loss.backward(retain_graph=True)
                        optimizerC_pre.step()
                

                D_loss.backward(retain_graph=True)
                optimizerD.step()
                
                G_loss.backward()
                optimizerG.step()
#                 print(G_loss)
#                 print(G_loss.item())
                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())
                C_losses.append(C_loss.item())

                
#             print(G_losses)
            G_epoch_loss.append(sum(G_losses) / len(G_losses))
            D_epoch_loss.append(sum(D_losses) / len(D_losses))
            C_epoch_loss.append(sum(C_losses) / len(C_losses))
#             D_epoch_acc.append(sum(D_accs) / len(D_accs))


        return C.state_dict(), G.state_dict(), D.state_dict(), sum(G_epoch_loss) / len(G_epoch_loss), sum(D_epoch_loss) / len(D_epoch_loss), sum(C_epoch_loss) / len(C_epoch_loss)
    

    
   
    
    def fine_tune(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):
                 # print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(img), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class server_finetune(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
#         self.selected_clients = []
        self.ldr = DataLoader(dataset,batch_size=self.args.local_bs_label,shuffle=True)
    
    def train(self, C, G, D, args, img_size):
        G.train()
        D.train()
        C.train()
        # train and update

        optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerC = torch.optim.Adam(D.parameters(), lr=3*args.lr, betas=(0.9, 0.999), eps=1e-08)
        optimizerC_pre = torch.optim.SGD(filter(lambda p: p.requires_grad, C.parameters()), lr=0.01, momentum=0.5)

        # Loss functions
        BCE_loss = torch.nn.BCELoss().cuda()
        CE_loss = self.loss_func
        
        one = one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        
        one = one.cuda()
        mone = mone.cuda()
        
        # label preprocess
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1


        G_epoch_loss = []
        D_epoch_loss = []
        C_epoch_loss = []
        

        for epoch in range(self.args.local_ep):
            D_losses = []
            G_losses = []
            C_losses = []
            C_preloss = []            
            
            for batch_idx, (image,labels) in enumerate(self.ldr):

                udt_label = labels.to(self.args.device).long()
                
                udt_label_d = fill[udt_label].cuda()
                udt_label_g = onehot[udt_label].cuda()
               

                mini_batch = image.size()[0]
                

                alpha = 0.5
                alpha_cla_adv = 0.01
                alpha_p = 0.1  # 0.1, 0.03


                # Adversarial ground truths
                y_real = torch.ones(mini_batch).float()
                y_fake = torch.ones(mini_batch).float()
                
                y_real, y_fake = y_real * 0.9, y_fake * 0.1
                y_real, y_fake = y_real.to(self.args.device).float(), y_fake.to(self.args.device).float()

                z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
                
                # A Game with Three Players

                ######################
                # utilizing of updated label
                ######################
                
                img_g = G(z, udt_label_g)
                log_probsD_g = D(img_g, udt_label_d)
                Dis_weight = log_probsD_g.mean()
                

                D_loss_fake = (1 - alpha) * torch.mean(BCE_loss(log_probsD_g, y_fake))



                for p in D.parameters():
                    p.requires_grad = False  # to avoid computation
                G.zero_grad()
                C.zero_grad()
                
                log_probsC = C(img_g)

                Rp = torch.mean(self.loss_func(log_probsC, udt_label))
    
                D_loss = D_loss_fake
                udt_labelg = labels.to(self.args.device).float()
                G_loss = (1 - alpha) * torch.mean(BCE_loss(log_probsD_g, y_real))
                
#                 C_loss = Dis_weight * alpha_p * Rp
                C_loss = Rp
                
                for p in D.parameters():  # reset requires_grad
                    p.requires_grad = True
                
                D_loss.backward(retain_graph=True)
                optimizerD.step()
                
                C_loss.backward(retain_graph=True)
                optimizerC_pre.step()
                
                G_loss.backward()
                optimizerG.step()


                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())
                C_losses.append(C_loss.item())

                

            G_epoch_loss.append(sum(G_losses) / len(G_losses))
            D_epoch_loss.append(sum(D_losses) / len(D_losses))
            C_epoch_loss.append(sum(C_losses) / len(C_losses))



        return C.state_dict(), G.state_dict(), D.state_dict(), sum(G_epoch_loss) / len(G_epoch_loss), sum(D_epoch_loss) / len(D_epoch_loss), sum(C_epoch_loss) / len(C_epoch_loss)

