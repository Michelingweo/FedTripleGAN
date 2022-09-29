#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
from skimage import io, transform
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import itertools
import pickle
import imageio
import numpy as np
import random
from collections import OrderedDict

from models.Update import DatasetSplit

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,sample
from utils.options import args_parser
from models.Update import LocalUpdate,server_finetune
from models.Nets import MLP, CNNMnist, CNNCifar, weights_init,generator,discriminator
from models.Fed import FedAvg
from models.test import test_img, test_img_client
from list_txt.make_list import maskvector_init
from torch.autograd import Variable
import matplotlib.animation as animation
from IPython.display import HTML

class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('user_num:',args.num_users)
    print('label_rate:',args.label_rate)
    print('epochs:',args.epochs)
    print('dataset:',args.dataset)
    print('frac:',args.frac)
    
    # Set random seed for reproducibility
    manualSeed = 999
#     manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)



    # ----------------------------------------------Dataset Choice--------------------------------------------

    # load dataset and split users

    # dataset[0]: set of (img, label)  dataset[i][0]: No.i img  dataset[i][1]: No.i label
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)
        # sample users
        if args.iid:
            dict_users, dict_users_label, dict_users_unlabel = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            dict_users, dict_users_label, dict_users_unlabel = mnist_noniid(dataset_train, args.num_users, args.label_rate)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [RandomTranslateWithReflect(4),
             transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users, dict_users_label, dict_users_unlabel = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'svhn':
        transform_svhn = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_svhn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('../data/svhn', split='train', download=True, transform=transform_svhn)
        dataset_test = datasets.SVHN('../data/svhn', split='test', download=True, transform=transform_svhn_test)

        if args.iid:
            dict_users, dict_users_label, dict_users_unlabel = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            exit('Error: only consider IID setting in svhn')

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    img_size = img_size[1]

    # --------------------------------------------Net choice--------------------------------------------------------
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_C = CNNCifar(args).to(args.device)
        net_G = generator().to(args.device)
        net_D = discriminator().to(args.device)
        net_G.weight_init()
        net_D.weight_init()


    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_C = CNNMnist(args).to(args.device)
        # net_G = ModelG(args.nz).to(args.device)
        # net_D = ModelD().to(args.device)
        net_G = generator().to(args.device)
        net_D = discriminator().to(args.device)
        net_G.weight_init()
        net_D.weight_init()


    elif args.dataset == 'svhn':
        net_C = CNNCifar(args).to(args.device)
        net_G = generator().to(args.device)
        net_D = discriminator().to(args.device)
        net_G.weight_init()
        net_D.weight_init()


    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
#         net_C = MLP().to(args.device)
        net_G = generator().to(args.device)
        net_D = discriminator().to(args.device)
#         net_G.weight_init(mean=0.0, std=0.02)
#         net_D.weight_init(mean=0.0, std=0.02)
        net_G.weight_init()
        net_D.weight_init()


    else:
        exit('Error: unrecognized model')
    print(net_C)
    print(net_G)
    print(net_D)

    net_C.train()
    net_G.train()
    net_D.train()

    # copy weights
    w_C = net_C.state_dict()
    w_G = net_G.state_dict()
    w_D = net_D.state_dict()

    # mask_vector initialization
    # 0: 1-labeled 0:pseudo-label -1-unlabeled 1:user's belonging 2: label/pseudo-label  3: data idx
    mask_vector, label_refer = maskvector_init(dataset_train, args, dict_users, dict_users_label)

    
#      # label preprocess
#     onehot = torch.zeros(10, 10)
#     onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
#     fill = torch.zeros([10, 10, img_size, img_size])
#     for i in range(10):
#         fill[i, i, :, :] = 1
#     z = torch.randn((args.local_bs_label, 100)).view(-1, 100, 1, 1)
# fixed noise & label
    temp_z_ = torch.randn(10, 100)
    fixed_z_ = temp_z_
    fixed_y_ = torch.zeros(10, 1)
    for i in range(9):
        fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
        temp = torch.ones(10, 1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)

    fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
    fix_noise = fixed_z_.to(args.device)
   
    fixed_y_label_ = torch.zeros(100, 10)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
    fix_label = fixed_y_label_.view(-1, 10, 1, 1).cuda()
    



    # training
    Closs_train = []
    Gloss_train = []
    Dloss_train = []
    
    img_list = []
#     fix_noise = torch.randn(64, 100, 1, 1).to(args.device)


    localGNet_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}  # 存储local Gnet所有layer的para
    localDNet_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}  # 存储local Dnet所有layer的para
    localCNet_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}  # 存储local Dnet所有layer的para
    localnum_dic = {idxs: 0 for idxs in range(args.num_users)}  # 存储local 出现频次
    m = max(int(args.frac * args.num_users), 1)

    for iter in range(args.epochs):
        Cw_locals,Gw_locals,Dw_locals, loss_Glocals, loss_Dlocals,loss_Clocals = [], [], [], [], [], []
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            net_D.load_state_dict(localDNet_dic[idx])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_label[idx],idxs2=dict_users_unlabel[idx],maskv=mask_vector)
            # w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            Cw, Gw, Dw, Gloss, Dloss, Closs = local.GAN_train(C=copy.deepcopy(net_C).to(args.device),G=net_G.to(args.device), D=copy.deepcopy(net_D).to(args.device),args=args, img_size=img_size, sampled_time=localnum_dic[idx])
            net_G.load_state_dict(Gw)
            # w_locals.append(copy.deepcopy(w))
            Gw_locals.append(copy.deepcopy(Gw))
            Dw_locals.append(copy.deepcopy(Dw))
            Cw_locals.append(copy.deepcopy(Cw))

            # Recording
            localnum_dic[idx]+=1
            localCNet_dic[idx] = copy.deepcopy(Cw)
            localGNet_dic[idx] = copy.deepcopy(Gw)
            localDNet_dic[idx] = copy.deepcopy(Dw)
            loss_Glocals.append(copy.deepcopy(Gloss))
            loss_Dlocals.append(copy.deepcopy(Dloss))
            loss_Clocals.append(copy.deepcopy(Closs))
            

#             label_fix = torch.tensor([0,1,2,3,4,5,6,7,8,9])
#             label_fix = onehot[label_fix].cuda()
            if (((iter*5*m)+idx*5) % 50 == 0):
                with torch.no_grad():
                    fake = net_G(fix_noise,fix_label).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # update global weights
        # w_glob = FedAvg(w_locals)
        # Gw_glob = FedAvg(Gw_locals)
        # Dw_glob = FedAvg(Dw_locals)
        Cw_glob = FedAvg(Cw_locals)

        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)
        # net_G.load_state_dict(Gw_glob)
        # net_D.load_state_dict(Dw_glob)
        net_C.load_state_dict(Cw_glob)

        # print loss
        Gloss_avg = sum(loss_Glocals) / len(loss_Glocals)
        Dloss_avg = sum(loss_Dlocals) / len(loss_Dlocals)
        Closs_avg = sum(loss_Clocals) / len(loss_Clocals)
        
        print('Round {:3d}, Average G loss {:.3f}'.format(iter, Gloss_avg))
        print('\tAverage D loss {:.3f}'.format(Dloss_avg))
        print('\tAverage C loss {:.3f}'.format(Closs_avg))
       
        Gloss_train.append(Gloss_avg)
        Dloss_train.append(Dloss_avg)
        Closs_train.append(Closs_avg)
        
        # testing
        net_C.eval()
        acc_train, loss_train = test_img(net_C, dataset_train, args)
        acc_test, loss_test = test_img(net_C, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        net_C.train()
         
#     for epoch in range(args.p1epochs):
#         Cw_locals,Gw_locals,Dw_locals, loss_Glocals, loss_Dlocals,loss_Clocals = [], [], [], [], [], []
#         fine_tune = server_finetune(args=args, dataset=dataset_train, maskv=mask_vector)
           
#         Cw, Gw, Dw, Gloss, Dloss, Closs = fine_tune.train(C=copy.deepcopy(net_C).to(args.device),G=copy.deepcopy(net_G).to(args.device), D=copy.deepcopy(net_D).to(args.device),args=args, img_size=img_size)
#         net_G.load_state_dict(Gw)
#         net_D.load_state_dict(Dw)
#         net_C.load_state_dict(Cw)
#         # Recording

#         loss_Glocals.append(copy.deepcopy(Gloss))
#         loss_Dlocals.append(copy.deepcopy(Dloss))
#         loss_Clocals.append(copy.deepcopy(Closs))
            
#         if ((epoch*20) % 50 == 0):
#             with torch.no_grad():
#                 fake = net_G(fix_noise,fix_label).detach().cpu()
#             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    

#         # print loss
#         Gloss_avg = sum(loss_Glocals) / len(loss_Glocals)
#         Dloss_avg = sum(loss_Dlocals) / len(loss_Dlocals)
#         Closs_avg = sum(loss_Clocals) / len(loss_Clocals)
        
#         print('FT Round {:3d}, \nAverage G loss {:.3f}'.format(epoch, Gloss_avg))
#         print('Average D loss {:.3f}'.format(Dloss_avg))
#         print('Average C loss {:.3f}'.format(Closs_avg))
       
#         Gloss_train.append(Gloss_avg)
#         Dloss_train.append(Dloss_avg)
#         Closs_train.append(Closs_avg)
#     # finetune round testing
#         net_C.eval()
#         acc_train, loss_train = test_img(net_C, dataset_train, args)
#         acc_test, loss_test = test_img(net_C, dataset_test, args)
#         print("Fine tune Training accuracy: {:.2f}".format(acc_train))
#         print("Fine tune Testing accuracy: {:.2f}".format(acc_test))
#         net_C.train()
    
    
    
    # plot loss curve
    plt.figure()
    plt.plot(range(len(Gloss_train)), Gloss_train,color='red', label='Generator loss')
    plt.plot(range(len(Dloss_train)), Dloss_train,color='blue', label='Discriminator loss')
    plt.plot(range(len(Closs_train)), Closs_train,color='green', label='Classifier loss')
    plt.xlabel('rounds')
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_user{}_ep{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.epochs, args.label_rate, args.iid))

    
#     # testing
#     net_D.eval()
#     acc_train, loss_train = test_img(net_D, dataset_train, args)
#     acc_test, loss_test = test_img(net_D, dataset_test, args)
#     print("Training accuracy: {:.2f}".format(acc_train))
#     print("Testing accuracy: {:.2f}".format(acc_test))


    #%%capture
    fig = plt.figure(figsize=(10,10))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    dataloader = DataLoader(dataset_train, batch_size=100,shuffle=True)
    # Grab a batch of real images from the dataloader
    for i, (img, label) in enumerate(dataloader):
        if i == 0:
            real_batch = img
            break

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(args.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('./save/real&fake_{}_user{}_ep{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.epochs, args.label_rate, args.iid))
    plt.show()
    

    