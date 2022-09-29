#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random



def maskvector_init(dataset,args,dicuser,dicuser_label):
    mask_vector = np.zeros((int(dataset.__len__()), 4),
                           dtype=int)  # 0: 1-labeled -1-unlabeled 1:user's belonging 2: label/pseudo-label  3: data idx
#     unlabel_vector = []
#     label_vector = []
    label_refer = []
    for i in range(dataset.__len__()):
        mask_vector[i][2] = dataset[i][1]
        label_refer.append(dataset[i][1])
        mask_vector[i][0] = 1
        mask_vector[i][3] = i
    label_idx = []
    for key in dicuser_label.keys():
        for labelidx in dicuser_label[key]:
            label_idx.append(labelidx)
    all_idx = np.arange(dataset.__len__())
    unlabel_idx = list(set(all_idx) - set(label_idx))
    for id in unlabel_idx:
        mask_vector[id][0] = -1
        mask_vector[id][2] = -1

    # user id assign
    for user_id in dicuser.keys():
        for id in dicuser[user_id]:
            mask_vector[id][1] = user_id
#     for i in range(len(mask_vector)):
#         if mask_vector[i][0] == 1:
#             label_vector[mask_vector[i][3]] = mask_vector[i]
#         elif mask_vector[i][0] == -1:
#             unlabel_vector[mask_vector[i][3]] = mask_vector[i]
    # return  mask_vector, label_vector, unlabel_vector
    return  mask_vector, label_refer


def dictrainUpdate(dic_user_label,dic_pseudo,arg):
    for i in range(arg.num_users):
        dic_user_label[i]+=dic_pseudo[i]

    return dic_user_label



