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
from models.Update_syn import server_finetune
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
    
    print('epochs:',args.server_epochs)
    print('dataset:',args.dataset)
    # Set random seed for reproducibility
#     manualSeed = 1070
    manualSeed = random.randint(1, 10000) # use if you want new results
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
            dict_users, dict_users_label = mnist_noniid(dataset_train, args.num_users, args.label_rate)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
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
        net_D=torch.load('./save_net/discriminator/Dmodel_cifar.pkl')
        net_C=torch.load('./save_net/classifer/Cmodel_cifar.pkl')
        net_G=torch.load('./save_net/generator/Gmodel_cifar.pkl')


    elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_D=torch.load('./save_net/discriminator/Dmodel_mnist.pkl')
#         net_C=torch.load('./save_net/classifer/Cmodel_mnist.pkl')
#         net_G=torch.load('./save_net/generator/Gmodel_mnist.pkl')
        net_C = CNNMnist(args).to(args.device)
        net_G = generator().to(args.device)
        net_D = discriminator().to(args.device)
        net_G.weight_init()
        net_D.weight_init()


    elif args.dataset == 'svhn':
        net_D=torch.load('./save_net/discriminator/Dmodel_svhn.pkl')
        net_C=torch.load('./save_net/classifer/Cmodel_svhn.pkl')
        net_G=torch.load('./save_net/generator/Gmodel_svhn.pkl')


    elif args.model == 'mlp':
        net_D=torch.load('./save_net/discriminator/Dmodel_mlp.pkl')
        net_C=torch.load('./save_net/classifer/Cmodel_mlp.pkl')
        net_G=torch.load('./save_net/generator/Gmodel_mlp.pkl')


    else:
        exit('Error: unrecognized model')
    print(net_C)
    print(net_G)
    print(net_D)

    net_C.train()
    net_G.train()
    net_D.train()


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
    img_list = []
    
    # waste train key
    acc_count = []
    wst_count = 0
    
    
    for iter in range(args.server_epochs):

        ft = server_finetune(args=args, dataset=dataset_train,idxs=dict_users_label[0])
        Cw, Closs = ft.train(C=copy.deepcopy(net_C).to(args.device),G=copy.deepcopy(net_G).to(args.device), D=copy.deepcopy(net_D).to(args.device),args=args, img_size=img_size)
            
            
        if ((iter*5) % 2 == 0):
            with torch.no_grad():
                fake = net_G(fix_noise,fix_label).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # copy weight to net_glob
        net_C.load_state_dict(Cw)

        # print loss
        
        print('Round {:3d}, FT C loss {:.3f}'.format(iter, Closs))
       

        Closs_train.append(Closs)
        
        # testing
        net_C.eval()
        acc_train, loss_train = test_img(net_C, dataset_train, args)
        acc_test, loss_test = test_img(net_C, dataset_test, args)
        acc_count.append(acc_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        net_C.train()
        
        # early stopping
        if acc_test < max(acc_count):
            wst_count+=1
        if wst_count >10:
            print("It is a waste to countinue to train.\n The training will be break.")
            break
            
    # plot loss curve
    plt.figure()
    plt.plot(range(len(Closs_train)), Closs_train,color='red', label='Classifier loss')
    plt.xlabel('rounds')
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_user{}_ep{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.epochs, args.label_rate, args.iid))

    # save model
    if args.model == 'cnn' and args.dataset == 'cifar':
        torch.save(net_D,'./save_net/discriminator/Dmodel_cifarft.pkl')
        torch.save(net_C,'./save_net/classifer/Cmodel_cifarft.pkl')
        torch.save(net_G,'./save_net/generator/Gmodel_cifarft.pkl')


    elif args.model == 'cnn' and args.dataset == 'mnist':

        torch.save(net_D.state_dict(),'/mnt/FLtripleGAN/save_net/discriminator/net_Dft.pkl')
        torch.save(net_C.state_dict(),'/mnt/FLtripleGAN/save_net/classifier/net_Cft.pkl')
        torch.save(net_G.state_dict(),'/mnt/FLtripleGAN/save_net/generator/net_Gft.pkl')

    elif args.dataset == 'svhn':
        torch.save(net_D,'./save_net/discriminator/Dmodel_svhnft.pkl')
        torch.save(net_C,'./save_net/classifer/Cmodel_svhnft.pkl')
        torch.save(net_G,'./save_net/generator/Gmodel_svhnft.pkl')


    elif args.model == 'mlp':
        torch.save(net_D,'./save_net/discriminator/Dmodel_mlpft.pkl')
        torch.save(net_C,'./save_net/classifer/Cmodel_svhn_mlpft.pkl')
        torch.save(net_G,'./save_net/generator/Gmodel_mlpft.pkl')
        
        
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
    

    