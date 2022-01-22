from __future__ import division
import os
import time
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from numpy.linalg import norm

from loss import CSQLoss,CSQCrossLoss
from utils import compute_eer,getRandPerson,one_hot_embedding,hamming_sim,cossim



def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def compute_result_prints_iitd(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            rbn = img[0].to(device, dtype=torch.float)
            output1 = net(rbn)#0-R,1-B,2-B,3-R-B, 4--NIR
            FEATS_prints.append(output1.cpu().numpy())
            GT.append(img[1])

    FEATS_prints = np.concatenate(FEATS_prints)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT

def compute_result_prints_vein(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            rbn = img[0].permute(0, 3, 1, 2).to(device, dtype=torch.float)

            output1 = net(rbn[:,0:2,:,:])#0-R,1-B,2-B,3-R-B, 4--NIR
            output2 = net(rbn[:,3:,:,:])#0-R,1-B,2-B,3-R-B, 4--NIR

            FEATS_prints.append(output1.cpu().numpy())
            FEATS_vein.append(output2.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[1])

    FEATS_prints = np.concatenate(FEATS_prints)
    FEATS_vein = np.concatenate(FEATS_vein)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT

def compute_result_prints_vein_2path(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            rbn = img[0].permute(0, 3, 1, 2).to(device, dtype=torch.float)

            output1, output2 = net(rbn[:,0:2,:,:],rbn[:,3:,:,:])#0-R,1-B,2-B,3-R-B, 4--NIR

            FEATS_prints.append(output1.cpu().numpy())
            FEATS_vein.append(output2.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[1])

    FEATS_prints = np.concatenate(FEATS_prints)
    FEATS_vein = np.concatenate(FEATS_vein)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT
def compute_result_prints_vein_2path2(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            rbn = img[0].to(device, dtype=torch.float)
            rbn2 = img[1].to(device, dtype=torch.float)

            output1, output2 = net(rbn,rbn2)#0-R,1-B,2-B,3-R-B, 4--NIR

            FEATS_prints.append(output1.cpu().numpy())
            FEATS_vein.append(output2.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[2])

    FEATS_prints = np.concatenate(FEATS_prints)
    FEATS_vein = np.concatenate(FEATS_vein)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT

def compute_result_prints_vein_2path_iitd(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            rbn = img[0].to(device, dtype=torch.float)

            output1 = net(rbn)#0-R,1-B,2-B,3-R-B, 4--NIR

            FEATS_prints.append(output1.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[1])

    FEATS_prints = np.concatenate(FEATS_prints)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT

def compute_result_prints_vein_2path_TJ(dataloader, net, device):
    FEATS_prints,FEATS_vein,GT = [],[],[]
    net.eval()
    with torch.no_grad():
        for batch_idx, img in enumerate(dataloader):
            in1 = img[0].to(device, dtype=torch.float)
            in2 = img[1].to(device, dtype=torch.float)
            label = img[2].to(device)

            output1, output2 = net(in1,in2)#0-R,1-B,2-B,3-R-B, 4--NIR

            FEATS_prints.append(output1.cpu().numpy())
            FEATS_vein.append(output2.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[2])

    FEATS_prints = np.concatenate(FEATS_prints)
    FEATS_vein = np.concatenate(FEATS_vein)
    GT = np.concatenate(GT)

    return FEATS_prints, FEATS_vein, GT
def eval_eer_hashcenter(FEATS_prints_binay,criterion, clsses = 500, sampesCls = 4):
    totalsamples = clsses*sampesCls
    pred_scores = []
    gt_label = []
    for i in tqdm.tqdm(range(totalsamples)):#50 11
        targethash = criterion.label2center(np.expand_dims(one_hot_embedding(i//sampesCls, clsses), axis=0))>0
        targethash = np.squeeze(targethash.detach().cpu().numpy())
        a = hamming_sim(FEATS_prints_binay[i,:],targethash)
        pred_scores.append(a)
        gt_label.append(True)

        wronghash = criterion.label2center(np.expand_dims(one_hot_embedding(getRandPerson(exclude=i//sampesCls,totalcls=clsses), clsses), axis=0))>0
        wronghash = np.squeeze(wronghash.detach().cpu().numpy())
        impo = hamming_sim(FEATS_prints_binay[i,:],wronghash)
        pred_scores.append(impo)
        gt_label.append(False)
        
    pred_scores = np.array(pred_scores)
    gt_label = np.array(gt_label)
    eer = compute_eer(gt_label, pred_scores)
    return eer

def eval_eer_fusion_center(FEATS_prints_binay,FEATS_veins_binay,criterion, clsses = 500, sampesCls = 4):
    totalsamples = clsses*sampesCls
    pred_scores = []
    gt_label = []
    for i in tqdm.tqdm(range(totalsamples)):#50 11
        targethash = criterion.label2center(np.expand_dims(one_hot_embedding(i//sampesCls, clsses), axis=0))>0
        targethash = np.squeeze(targethash.detach().cpu().numpy())
        a1 = hamming_sim(FEATS_prints_binay[i,:],targethash)
        a2 = hamming_sim(FEATS_veins_binay[i,:],targethash)
        pred_scores.append((a1+a2)/2.0)
        gt_label.append(True)

        wronghash = criterion.label2center(np.expand_dims(one_hot_embedding(getRandPerson(exclude=i//sampesCls,totalcls=clsses), clsses), axis=0))>0
        wronghash = np.squeeze(wronghash.detach().cpu().numpy())
        impo1 = hamming_sim(FEATS_prints_binay[i,:],wronghash)
        impo2 = hamming_sim(FEATS_veins_binay[i,:],wronghash)
        pred_scores.append((impo1+impo2)/2.0)
        gt_label.append(False)
        
    pred_scores = np.array(pred_scores)
    gt_label = np.array(gt_label)
    eer = compute_eer(gt_label, pred_scores)
    return eer


### FEATS_prints_binay -> FEATS_prints
def eval_eer_fusion(FEATS_prints_binay,FEATS_veins_binay, clsses = 500, sampesCls = 4, disfun='cossim'):
    totalsamples = clsses*sampesCls
    pred_scores = []
    gt_label = []

    for i in range(totalsamples):
        for j in range(i+1,totalsamples):
            # pred_scores.append(final[i,j].detach().cpu().numpy())
            if disfun == 'cossim':
                a1 = cossim(FEATS_prints_binay[i,:],FEATS_prints_binay[j,:])
                a2 = cossim(FEATS_veins_binay[i,:],FEATS_veins_binay[j,:])
            else:
                a1 = hamming_sim(FEATS_prints_binay[i,:],FEATS_prints_binay[j,:])
                a2 = hamming_sim(FEATS_veins_binay[i,:],FEATS_veins_binay[j,:])
            pred_scores.append((a1+a2)/2.0)
            gt_label.append(i//sampesCls == j//sampesCls)

    pred_scores = np.array(pred_scores)
    gt_label = np.array(gt_label)

    eer = compute_eer(gt_label, pred_scores)
    return eer

def eval_eer(FEATS_prints, clsses = 500, sampesCls = 4, disfun='cossim'):
    totalsamples = clsses*sampesCls
    pred_scores = []
    gt_label = []

    for i in range(totalsamples):
        for j in range(i+1,totalsamples):
            # pred_scores.append(final[i,j].detach().cpu().numpy())
            if  disfun=='cossim':
                a1 = cossim(FEATS_prints[i,:],FEATS_prints[j,:])
                pred_scores.append(a1)
            else:
                a1 = hamming_sim(FEATS_prints[i,:],FEATS_prints[j,:])
                pred_scores.append(a1)
            gt_label.append(i//sampesCls == j//sampesCls)

    pred_scores = np.array(pred_scores)
    gt_label = np.array(gt_label)

    eer = compute_eer(gt_label, pred_scores)
    return eer

def eval_eer_cross(FEATS_prints,FEATS_veins, clsses = 500, sampesCls = 4, disfun=cossim):
    totalsamples = clsses*sampesCls
    pred_scores_cross = []
    gt_label_cross = []

    for i in tqdm.tqdm(range(totalsamples)):#50 11
        a = disfun(FEATS_prints[i,:],FEATS_veins[i,:])# RED vs NIR same
        pred_scores_cross.append(a)
        gt_label_cross.append(True)

        impo = disfun(FEATS_prints[i,:],FEATS_veins[getRandPerson(exclude=i//sampesCls,totalcls=clsses)*sampesCls,:])
        pred_scores_cross.append(impo)
        gt_label_cross.append(False)

    pred_scores_cross = np.array(pred_scores_cross)
    gt_label_cross = np.array(gt_label_cross)

    eer_cross = compute_eer(gt_label_cross, pred_scores_cross)
    return eer_cross

def eval_top1(FEATS_prints_binay,criterion, clsses = 500, sampesCls = 4):
    def bitwisexor(a,b):
        return norm(np.bitwise_xor(a,b), axis=1, ord=1)
    totalsamples = clsses*sampesCls
    targethashs = []
    top1 = 0
    for i in range(clsses):
        targethash = criterion.label2center(np.expand_dims(one_hot_embedding(i, clsses), axis=0))>0
        targethash = np.squeeze(targethash.detach().cpu().numpy())
        targethashs.append(targethash)
    targethashs = np.array(targethashs)
    acc = 0
    for i in tqdm.tqdm(range(totalsamples)):#50 11
        dist =  bitwisexor(FEATS_prints_binay[i,:],targethashs)
        minindex = np.argmin(dist)
        # print('minindex',minindex,' i ',i, dist[minindex])
        if minindex == i//sampesCls:
            acc = acc + 1
    acc = acc / totalsamples
    return acc

def eval_fusion_top1(FEATS_prints_binay,FEATS_veins_binay,criterion, clsses = 500, sampesCls = 4):
    def bitwisexor(a,b):
        return norm(np.bitwise_xor(a,b), axis=1, ord=1)
        
    totalsamples = clsses*sampesCls
    targethashs = []
    top1 = 0
    for i in range(clsses):
        targethash = criterion.label2center(np.expand_dims(one_hot_embedding(i, clsses), axis=0))>0
        targethash = np.squeeze(targethash.detach().cpu().numpy())
        targethashs.append(targethash)
    targethashs = np.array(targethashs)
    acc = 0
    for i in tqdm.tqdm(range(totalsamples)):#50 11
        dist =  bitwisexor(FEATS_prints_binay[i,:],targethashs)
        dist2 =  bitwisexor(FEATS_veins_binay[i,:],targethashs)
        minindex1 = np.argmin(dist)
        minindex2 = np.argmin(dist2)
        if minindex1 == minindex2:
            minindex = minindex1
        elif dist[minindex1] < dist2[minindex2]:
            minindex = minindex1
        else:
            minindex = minindex2

        if minindex == i//sampesCls:
            acc = acc + 1
    acc = acc / totalsamples
    return acc
