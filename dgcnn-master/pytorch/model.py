#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature



def get_PE(x,  indices, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = indices + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = (feature-x).permute(0, 3, 1, 2).contiguous()
  
    return feature


def get_NN_feature(x, indices, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)   
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = indices + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)     
    feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature



def get_score_feature(x, y, indices, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)   
    y = y.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = indices + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    y = y.transpose(2, 1).contiguous()
    feature = y.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = (feature-x).permute(0, 3, 1, 2).contiguous()

    return feature

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
class SAN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(SAN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        ####################conv_in ここから#################
        self.conv_in = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        ####################ここまで########################
        self.fc1 = nn.Conv1d(64, 64, kernel_size=1)
        self.sa1 = BottleNeck(64, 64//2)
        ####################################################
        ####################ここまで########################
        self.fc2 = nn.Conv1d(64, 64, kernel_size=1)
        self.sa2 = BottleNeck(64, 64//2)
        ####################################################
        ####################ここまで########################
        self.fc3 = nn.Conv1d(64, 128, kernel_size=1)
        self.sa3 = BottleNeck(128, 128//2)
        ####################################################
        ####################ここまで########################
        self.fc4 = nn.Conv1d(128, 256, kernel_size=1)
        self.sa4 = BottleNeck(256, 256//2)
        ####################################################
        self.fc5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.ReLU)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
    
    def forward(self, x):
        batch_size = x.size(0)
        xyz_position = x
        ##########conv_in################
        x = self.conv_in(x)
        #################################
        indices = knn(x, self.k)
        x = self.sa1( self.fc1(x), indices, xyz_position )
        x1 = x
        ##################################
        indices = knn(x, self.k)
        x = self.sa2( self.fc2(x), indices, xyz_position )
        x2 = x
        #################################
        indices = knn(x, self.k)
        x = self.sa3( self.fc3(x), indices, xyz_position )
        x3 = x
        #################################
        indices = knn(x, self.k)
        x = self.sa4( self.fc4(x), indices, xyz_position )
        x4 = x
        #################################
        x = torch.cat((x1, x2, x3, x4), dim=1)
        ################################
        x = self.fc5(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self, input_channel, mid_channel):
        super(BottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_channel)
        self.sam = SAM(input_channel, mid_channel)
        self.bn2 = nn.BatchNorm1d(mid_channel)
        self.conv = nn.Conv1d(mid_channel, input_channel, kernel_size=1)
            
    def forward(self, x, indices, xyz_position):
        identity = x
        out = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.sam(out, indices, xyz_position)))
        out = self.conv(out)
        out += identity
        return out

class SAM(nn.Module):
    def __init__(self, input_channel, mid_channel):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv1d(input_channel, mid_channel, kernel_size=1)
        self.conv3 = nn.Conv1d(input_channel, mid_channel, kernel_size=1)
        self.conv4 = nn.Conv1d(3, mid_channel, kernel_size=1)
        self.conv_score = nn.Conv2d(mid_channel * 2, mid_channel, kernel_size=1)
        self.conv_value = nn.Conv2d(mid_channel * 2, mid_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channel * 2)
        self.bn2 = nn.BatchNorm2d(mid_channel * 2)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, indices, xyz_position):
        query = self.conv1(x)
        key = self.conv2(x)
        value = self.conv3(x)
        position = self.conv4(xyz_position)
        
        k = indices.shape[-1]
        
        score = get_score_feature(query, key, indices, k)
        values = get_NN_feature(value, indices, k)
        positional =  get_PE(position,  indices, k)
        
        score = torch.cat((score, positional), dim=1)
        values = torch.cat((values, positional), dim=1)
        
        score = F.relu(self.bn1(score))
        values = F.relu(self.bn2(values))
        
        score = self.softmax(self.conv_score(score))
        values = self.conv_value(values)
        
        out = torch.mul(score, values)
        out = out.sum(dim=-1, keepdim=False)
        return out
