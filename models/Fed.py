#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import collections
from scipy.stats import wasserstein_distance as wd


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# def Fedrop(netw,avgw):
#     emd_dic = {}
    
#     wn = copy.deepcopy(netw)
#     wa = copy.deepcopy(avgw)
#     for key in wn.keys():
#         for i in range(1,len(wn)):
#             wn[key]=wn[key].cpu()
#             wa[key]=wa[key].cpu()
#             emd = wd(wn[key].flatten(),wa[key].flatten())
#             emd_dic[key] = emd
#     maxkey = max(emd_dic, key=emd_dic.get)
#     wn[maxkey] = wa[maxkey]
#     wn = wn
    
#     return wn
            
    

def Fedrop(Netw_dic,idx_list,avgw):
    '''
    Netw_dic: a dic which stores all the client net para
    idx_list: a list contains the client id who participate into this round
    avgw: the global net para of last server round
    '''
    
    w_locals = []
    wa = copy.deepcopy(avgw) 
#     idx_DivgDic = {}
#     for id in idx_list:
#         divg_dic = {}
#         wn = Netw_dic[id]
#         for key in wa.keys():
#             wn[key]=wn[key].cpu()
#             wa[key]=wa[key].cpu()
#             divg_w = ( (wn[key]-wa[key]).float().norm(2)**2  , wn[key].float().norm(2)**2  )
#             divg_dic[key] = divg_w
#         idx_DivgDic[id] = divg_dic
    for key in wa.keys():
        divg_list=[]
        for id in idx_list:
            wn = Netw_dic[id]
            wn[key]=wn[key].cpu()
            wa[key]=wa[key].cpu()
            divg_w = ( (wn[key]-wa[key]).float().norm(2)**2)  / (wn[key].float().norm(2)**2)
#             divg_w = ( (wn[key]-wa[key]).float().norm(2)**2, wn[key].float().norm(2)**2)
            divg_list.append(divg_w)
        _ = divg_list.index(max(divg_list))
        drop_idx =idx_list[_]
        Netw_dic[drop_idx][key] = wa[key]
    for id in idx_list:
        w_locals.append(Netw_dic[id])
    
    return Netw_dic, w_locals


def WeightDisGet(Netw_dic,idx_list,avgw):
    '''
    Netw_dic: a dic which stores all the client net para
    idx_list: a list contains the client id who participate into this round
    avgw: the global net para of last server round
    '''
    
    w_locals = []
    wa = copy.deepcopy(avgw) 
    idx_DisDic = {}
    for id in idx_list:
        dis_dic = []
        wn = Netw_dic[id]
        for key in wa.keys():
#             print(key)
            key = str(key)
            wn[key]=wn[key].cpu()
            wa[key]=wa[key].cpu()
            dis_w = ( (wn[key]-wa[key]).float().norm(2)**2)
            dis_dic.append(dis_w)
        idx_DisDic[id] = dis_dic.mean()
    return idx_DisDic


def FedReplace(Netw_dic, idx_list, avgw):
    w_locals = []
    wa = copy.deepcopy(avgw)
    idx_Divg = {}
    
    for id in idx_list:
        wn = Netw_dic[id]
        diff_list = {}
        for key in wa.keys():
            wn[key] = wn[key].cpu()
            wa[key] = wa[key].cpu()
            # divg_w = ((wn[key] - wa[key]).float().norm(2) ** 2)/ (wn[key].float().norm(2) ** 2)
            pos_negClient = np.sign(wn[key])
            pos_negGlob = np.sign(wa[key])
            p_n = pos_negClient + pos_negGlob
#             print(p_n.size())
            
            uniq,count = np.unique(p_n, return_counts=True)
            stat_dic = dict(zip(uniq, count))
            similar_count = 0
            if (2 in stat_dic.keys()):
                similar_count = stat_dic[2]
            if (-2 in stat_dic.keys()):
                similar_count+=stat_dic[-2]
            total_count=0
            for key_s in stat_dic.keys():
                total_count += stat_dic[key_s]
            similarity = similar_count/total_count
            if similarity < 0.8:
                Netw_dic[id][key] = wa[key]
                continue
            replace_index = np.where(p_n == 0)
            for ridx in range(len(replace_index[0])):
                l_ = len(replace_index)
#                 print(key)
#                 print(replace_index[ridx])
#                 print(wn[key].size())
                if l_ == 4:
                    wn[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])][int(replace_index[2][ridx])][int(replace_index[3][ridx])] =  wa[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])][int(replace_index[2][ridx])][int(replace_index[3][ridx])] 
                elif l_ == 3:
                    wn[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])][int(replace_index[2][ridx])] = \
                    wa[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])][int(replace_index[2][ridx])]
                elif l_ == 2:
                    wn[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])] = \
                        wa[key][int(replace_index[0][ridx])][int(replace_index[1][ridx])]
                elif l_ == 1:
                    wn[key][int(replace_index[0][ridx])] = \
                        wa[key][int(replace_index[0][ridx])]
        Netw_dic[id] = wn

    for id in idx_list:
        w_locals.append(Netw_dic[id])

    return Netw_dic, w_locals


def Dua_Comp(Doutput_dic,wclass,dr):
