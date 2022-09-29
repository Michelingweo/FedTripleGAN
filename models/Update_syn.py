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


def distrib2odds(Dx):
    odds = Dx/(1-Dx)
    return odds

def odds2distrib(odds):
    Dx = odds/(1+odds)
    return Dx


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

    


class LocalUpdate(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.selected_clients = []

        self.ldr_label = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs_label, shuffle=True)
#         self.ldr_label = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs_label, shuffle=True)
        self.ldr_unlabel = DataLoader(DatasetSplit_mask(dataset, idxs2, mask_vector=maskv),batch_size=self.args.local_bs_unlabel, shuffle=True)

    def local_train(self, C, G, D, args, img_size, sampled_time,data_ratio,Gimg):
        G.eval()
        D.train()
        C.train()
        # train and update


        optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerC = torch.optim.Adam(C.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08)
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


        D_epoch_output = []
        D_epoch_loss = []
        C_epoch_loss = []
        
        alpha_P = 0.5
        alpha_pseudo = 0.1
        alpha_threshold = 200

        for epoch in range(self.args.local_ep):
            D_losses = []
            D_ouput = []
            C_losses = []

            
            
            
            
            for batch_idx, ((image1, labels1),(image2,labels2)) in enumerate(zip(cycle(self.ldr_label),self.ldr_unlabel)):

                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()
                image_u, label_u = image2.to(self.args.device), labels2.to(self.args.device).long()
                label_ld = fill[label_l].cuda()
                label_ud = fill[label_u].cuda()
                label_lg = onehot[label_l].cuda()
                label_ug = onehot[label_u].cuda()

                mini_batch_l = image_l.size()[0]
                mini_batch_u = image_u.size()[0]

                # Adversarial ground truths
                y_real_l = torch.ones(mini_batch_l).float()
                y_fake_l = torch.ones(mini_batch_l).float()
                
                y_real_l, y_fake_l = y_real_l * 0.9, y_fake_l * 0.1
                y_real_l, y_fake_l = y_real_l.to(self.args.device).float(), y_fake_l.to(self.args.device).float()

                z_l = torch.randn((mini_batch_l, 100)).view(-1, 100, 1, 1).cuda()
#                 z_l = z_l.long()

                # Adversarial ground truths
                y_real_u = torch.ones(mini_batch_u).float()
                y_fake_u = torch.ones(mini_batch_u).float()
                # label reverse
                # y_fake = torch.ones(mini_batch).float()
                # y_real = torch.ones(mini_batch).float()
                y_real_u, y_fake_u = y_real_u * 0.9, y_fake_u * 0.1
                y_real_u, y_fake_u = y_real_u.to(self.args.device).float(), y_fake_u.to(self.args.device).float()

                z_u = torch.randn((mini_batch_u, 100)).view(-1, 100, 1, 1).cuda()
#                 z_u = z_u.long()

                # A Game with Three Players

                ######################
                # train Discriminator by labeled data
                ######################
                D.zero_grad()
                log_probsD_real = D(image_l, label_ld)
                D_loss_real = torch.mean(BCE_loss(log_probsD_real, y_real_l)) # Mean(log(D(x,y)))


                ######################
                # train Generator by labeled data
                ######################
                G.zero_grad()
                img_g = G(z_l, label_lg)
                log_probsD_g = D(img_g, label_ld)
                odds = distrib2odds(log_probsD_g)
                Dx = odds2distrib(odds * data_ratio)
                D_ouput.append(Dx)

                D_loss_fake = torch.mean(BCE_loss(log_probsD_g, y_fake_l)) # (1-alpha)*Mean(log(1-D(x_g,y)))


                ######################
                # train Classifer C by labeled data
                ######################
                C.zero_grad()
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
                pseudo_labeld = fill[_].cuda()
                log_probsD_fake = D(image_u, pseudo_labeld) 

                D_loss_cla = torch.mean(BCE_loss(log_probsD_fake, y_fake_u))# alpha*Mean(log(1-D(x,y_c)))

#                 C_loss_dis = torch.mean(max_c * self.loss_func(log_probsD_fake, y_real_u))
                C_loss_dis = torch.mean(p_c*BCE_loss1(log_probsD_fake, y_real_u))

                log_probsC = C(img_g)
#                 pseudo_label_g = torch.argmax(log_probsC, dim=1)
                Rp = torch.mean(self.loss_func(log_probsC, label_l)) # Rp

        
                D_loss = D_loss_real + (1 - alpha_P) * D_loss_fake + alpha_P * D_loss_cla

            
            
                if sampled_time * 10 > alpha_threshold:
#                     C_loss = 0.01 * alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp
                    C_loss = alpha_P * C_loss_dis + Rl + alpha_pseudo *Rp
                    C_loss.backward(retain_graph=True)
                    optimizerC.step()
                elif sampled_time * 10 <=alpha_threshold and sampled_time * 10 >= 100:
#                     C_loss =  0.01 * alpha_P * C_loss_dis + Rl 
                    C_loss = alpha_P * C_loss_dis + Rl
                    C_loss.backward(retain_graph=True)
                    optimizerC.step()
                else: 
                    C_loss = Rl
                    C_loss.backward(retain_graph=True)
                    optimizerC_pre.step()

                D_loss.backward(retain_graph=True)
                optimizerD.step()
                D_losses.append(D_loss.item())
                C_losses.append(C_loss.item())

            D_epoch_output.append(D_ouput)
            D_epoch_loss.append(sum(D_losses) / len(D_losses))
            C_epoch_loss.append(sum(C_losses) / len(C_losses))



        return C.state_dict(), D.state_dict(), D_epoch_output, sum(D_epoch_loss) / len(D_epoch_loss), sum(C_epoch_loss) / len(C_epoch_loss)

    
   
    

class server_train(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None, idx_all=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.selected_clients = []
        self.id1 = idxs
        self.id2 = idxs2
        self.ldr_label = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv),
                                    batch_size=self.args.local_bs_label, shuffle=True)
        #         self.ldr_label = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs_label, shuffle=True)
        self.ldr_unlabel = DataLoader(DatasetSplit_mask(dataset, idxs2, mask_vector=maskv),
                                      batch_size=self.args.local_bs_unlabel, shuffle=True)
        self.ldr = DataLoader(DatasetSplit_mask(dataset, idxs2, mask_vector=maskv),
                                      batch_size=self.args.local_bs_unlabel, shuffle=True)

    def G_generation(self, G, args, img_size):
        G.eval()
        # label preprocess
        Gimg_dic = {}

        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1

        length_u = int(self.id2/args.local_bs_unlabel)+1
        length_l = int(self.id1/args.local_bs_unlabel)+1

        for epoch in range(self.args.local_ep):
            Gimg_epoch = []
            for batch_idx,(image1, labels1) in enumerate(range(max(length_u,length_l)),self.ldr):

                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()

                label_ld = fill[label_l].cuda()
                label_lg = onehot[label_l].cuda()

                mini_batch_l = image_l.size()[0]
                z_l = torch.randn((mini_batch_l, 100)).view(-1, 100, 1, 1).cuda()

                img_g = G(z_l, label_lg)
                Gimg_epoch.append((img_g,label_ld))

            Gimg_dic[epoch] = Gimg_epoch

        return Gimg_dic


    def G_train(self, G,args,img_size,Dua):
        G.train()
        # train and update
        # train and update

        optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Loss functions
        BCE_loss = torch.nn.BCELoss().cuda()

        # label preprocess
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1

        G_epoch_loss = []
        alpha_P = 0.5
        for epoch in range(self.args.local_ep):
            G_losses = []

            for batch_idx, ((image1, labels1), (image2, labels2)) in enumerate(
                    zip(cycle(self.ldr_label), self.ldr_unlabel)):

                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()

                mini_batch_l = image_l.size()[0]


                # Adversarial ground truths
                y_real_l = torch.ones(mini_batch_l).float()
                y_fake_l = torch.zeros(mini_batch_l).float()
                y_real_l, y_fake_l = y_real_l.to(self.args.device).float(), y_fake_l.to(self.args.device).float()

                ######################
                # train Generator by label
                ######################
                G.zero_grad()

                log_probsD_g = Dua[epoch][batch_idx]

                G_loss = (1 - alpha_P) * torch.mean(BCE_loss(log_probsD_g, y_fake_l))
                G_loss.backward()
                optimizerG.step()
                G_losses.append(G_loss.item())


            G_epoch_loss.append(sum(G_losses) / len(G_losses))


        return  G.state_dict(),  sum(G_epoch_loss) / len(G_epoch_loss)

