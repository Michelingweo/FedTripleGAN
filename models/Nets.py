#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from utils.options import args_parser
import numpy as np

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def custom_activation(output):
    logexpsum = torch.sum(torch.exp(output),axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# class ModelD(nn.Module):
#     def __init__(self):
#         super(ModelD, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.fc1  = nn.Linear(64*28*28+1000, 1024)
#         self.fc2 = nn.Linear(1024, 1)
#         self.fc3 = nn.Linear(10, 1000)
#
#     # weight_init
#     def weight_init(self):
#         weights_init(self._modules)
#
#     def forward(self, x, labels):
#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, 28,28)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = x.view(batch_size, 64*28*28)
#         y_ = self.fc3(labels)
#         y_ = F.relu(y_)
#         x = torch.cat([x, y_], 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return F.sigmoid(x)

class ModelG(nn.Module):
    def __init__(self, z_dim=100):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim+1000, 64*28*28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    # weight_init
    def weight_init(self):
        weights_init(self._modules)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x

# class generator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(generator, self).__init__()
#         self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
#         self.deconv1_1_bn = nn.BatchNorm2d(d*2)
#         self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
#         self.deconv1_2_bn = nn.BatchNorm2d(d*2)
#         self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
#         self.deconv2_bn = nn.BatchNorm2d(d*2)
#         self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
#         self.deconv3_bn = nn.BatchNorm2d(d)
#         self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input, label):
#         x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
#         y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
#         x = torch.cat([x, y], 1)
#         x = F.relu(self.deconv2_bn(self.deconv2(x)))
#         x = F.relu(self.deconv3_bn(self.deconv3(x)))
#         x = F.tanh(self.deconv4(x))
#         # x = F.relu(self.deconv4_bn(self.deconv4(x)))
#         # x = F.tanh(self.deconv5(x))

#         return x


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, ):
        super(generator, self).__init__()
#         self.fc2 = nn.Linear(10, 1000)
#         self.fc = nn.Linear(100 + 1000, 64 * 32 * 32)
#         self.bn1 = nn.BatchNorm2d(64)
        self.deconv1_1 = nn.ConvTranspose2d(100, 128*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(128*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, 128*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(128*2)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(128*4, 128*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128*2),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(128*2, 128, 4, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(128, args.num_channels, 4, 2, 1, bias=False),

            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def weight_init(self):
        weights_init(self._modules)
    # forward method
    def forward(self, z, label):
        batch_size = z.size(0)
#         y_ = self.fc2(label)
#         y_ = F.relu(y_)
#         x = torch.cat([x, y], 1)
#         x = self.fc(x)
#         x = x.view(batch_size, 64, 28, 28)
#         x = self.bn1(x)
#         x = F.relu(x)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        return self.main(x)

# class discriminator(nn.Module):
#     def __init__(self, d=128):
#         super(discriminator, self).__init__()
#         self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1,bias=False)
#         self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1,bias=False)
#         self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1,bias=False)
#         self.conv2_bn = nn.BatchNorm2d(d*2)
#         self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1,bias=False)
#         self.conv3_bn = nn.BatchNorm2d(d*4)
#         self.conv4 =nn.Sequential(
#             nn.Conv2d(d * 4, 128, 4, 1, 0,bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
#             nn.BatchNorm2d(128, 0.8)
#         )

#     # weight_init
#     def weight_init(self):
#         weights_init(self._modules)
#     # forward method
#     def forward(self, input, label):
#         x = F.leaky_relu(self.conv1_1(input), 0.2,inplace=True)
#         y = F.leaky_relu(self.conv1_2(label), 0.2,inplace=True)
#         x = torch.cat([x, y], 1)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2,inplace=True)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2,inplace=True)
#         x = self.conv4(x)
#         x = x.view(x.size(0),-1)

#         validity = custom_activation(out)# 1/0

#         return validity
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(args.num_channels, 64, 4, 2, 1, bias=False)
        self.conv1_2 = nn.Conv2d(10, 64, 4, 2, 1, bias=False)

        self.conv_blocks = nn.Sequential(

#             nn.Conv2d(args.num_channels, 128, 4, 2, 1,bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),

            nn.Conv2d(128, 128*2, 4, 2, 1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128*2, 0.8),

            nn.Conv2d(128*2, 128*4, 4, 2, 1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128*4, 0.8),

            nn.Conv2d(128*4, 128, 4, 1, 0,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8)
        )

        # Output layers
        self.adv_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.aux_layer = nn.Sequential(
            nn.Linear(128, args.num_classes),
            nn.Softmax()
        )
    # weight_init
    def weight_init(self):
        weights_init(self._modules)

    def forward(self, img, label):
#         print(label)
        x = F.leaky_relu(self.conv1_1(img), 0.2, inplace=True)
        y = F.leaky_relu(self.conv1_2(label), 0.2, inplace=True)
#         print(x.size())
#         print(y.size())
        input = torch.cat([x, y], 1)
        out = self.conv_blocks(input)
#         print(out.size())
        out = out.view(out.size(0),-1)
#         print(out.size())
#         validity = self.adv_layer(out)
        validity = custom_activation(out)# 1/0
        # label = self.aux_layer(out) # softmax

        # return validity, label
        return validity

    
    
    
# Classifier Network
    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNNCifar(nn.Module):
    """CNN."""

    def __init__(self, args):
        """CNN Builder."""
        super(CNNCifar, self).__init__()

        # Conv Layer block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=args.num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_layer = nn.Sequential(

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""

        x = self.conv1(x)
        x = self.conv2(x)

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

