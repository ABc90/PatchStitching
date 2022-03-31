#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Jun Zhou
@Time: 2020/10/29
The code is modified from Pcpnet and
the feature learn part is modified from DGCNN (@Author: Yue Wang)
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pcpnet_utils as utils
# We copied the TransformerEncoder, TransformerEncoderLayer and MultiheadAttention code from pytorch 1.3 code base
# so we can run with PyTorch >= 1.1.
from transplant_attn.transformer_from_torch import TransformerEncoder, TransformerEncoderLayer
from transplant_attn.MultiheadAttention_from_torch import MultiheadAttention



def knn(x, k,sort=True):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1,sorted=sort)[1]  # (batch_size, num_points, k)
    return idx

def knn2(x,y,k,sort=True):
    inner = -2 * torch.matmul(x.transpose(2, 1), y)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1, sorted=sort)[1]  # (batch_size, num_points, k)
    return idx


def run_attn(attn, x, use_ffn):
    """
    :param attn:        Attention functions, currently support nn.Multihead and TransformerEncoder
    :param x:           Input embeddings in shape (B, K, N, C), C denotes the No. of channels for each point, e.g. 512.
    :param use_ffn:     if choose use_ffn, the input is only x
    :return:            Soft attn output () and weights. The returned weights is None if use TransformerEncoder
    """
    attn_out_list = []
    weights_list = []
    weights = None
    B = x.shape[0]

    if use_ffn:
        for b in range(B):
            attn_out, weights_t = attn(x[b])  # x: (K, N, 512), attn_out: (K, N, 512), weights: (N, K, K)
            attn_out_list.append(attn_out)
            weights_list.append(weights_t)
    else:
        for b in range(B):
            attn_out, weights_t = attn(x[b], x[b], x[b])  # x: (K, N, 512), attn_out: (K, N, 512), weights: (N, K, K)
            attn_out_list.append(attn_out)
            weights_list.append(weights_t)

    # The weights are only for debug and visualisation.
    # We just return the (N, K) matrix, not the full (N, K, K) tensor.
    # if weights is not None:
    #     weights = weights[:, 0, :]
    # else:
    weights = torch.stack(weights_list)

    x = torch.stack(attn_out_list)
    return x, weights


def get_graph_feature(opt, x, k=20, d = 1, idx=None):
    """
    :param x: B*C*N C=3 if just points are given
    :param k:
    :param idx:
    :return:
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_dilated = knn(x, k=k*d)  # (batch_size, num_points, k)
        idx = idx_dilated[:,:,::d]

    # device = torch.device('cuda')
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)


    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # (feature - x, x) is not trans invariant, maybe just use relative features?
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # feature = torch.cat((feature - x, feature - x), dim=3).permute(0, 3, 1, 2).contiguous()
    # feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature





class Point_Shitf_Net(nn.Module):
    def __init__(self):
        super(Point_Shitf_Net, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)


        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=True),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=True),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        ## obtain a local translate of each point
        self.deconv1 = nn.Sequential(nn.Conv1d(256*2, 128, kernel_size=1, bias=True),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.deconv2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=True),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.deconv3 = nn.Conv1d(64, 3, kernel_size=1, bias=True)  ## we do not use activation last layer

    def forward(self, x):
        num_points = x.size(2)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 256, num_points)
        x1 = x.max(dim=-1, keepdim=True)[0]  # (batch_size, 256, num_points) -> (batch_size, 256,1)
        x = torch.cat((x,x1.repeat(1,1,num_points)),dim=1)
        x = self.deconv1(x)    #(batch_size, 512, num_points) -> (batch_size,64,num_points)
        x = self.deconv2(x)    #(batch_size, 128, num_points) -> (batch_size,64,num_points)
        x = self.deconv3(x)    #(batch_size, 64, num_points) -> (batch_size,3,num_points)

        return x


class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=True),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=True),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=True)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 4)
        # init.constant_(self.transform.weight, 0)
        # init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 4)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden
        # convert quaternion to rotation matrix
        x = utils.batch_quat_to_rotmat(x)
        return x


class ExpertsNet(nn.Module):
    def __init__(self):
        super(ExpertsNet, self).__init__()

        self.scales = 3
        self.input_dim = 64 * 3

        self.fc_layer1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.fc_layer2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2))
        self.fc_layer3 = nn.Sequential(nn.Conv1d(32, self.scales, kernel_size=1, bias=False))
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.cuda.FloatTensor):
        v = self.fc_layer1(x)
        v = self.fc_layer2(v)
        v = self.fc_layer3(v)
        v = self.sm(v)
        return v


class PointFeatureNet(nn.Module):
    """
    This part is mainly based on DGCNN for extracting features of each points from
    a local patch we use 1024/2048 points as inputs of the DGCNN
    """

    def __init__(self,opt, input_channels=3,k=20,use_point_stn=True,use_point_shift = True):
        super(PointFeatureNet,self).__init__()
        self.k = k
        self.opt = opt
        c_in = input_channels
        self.use_point_stn = use_point_stn
        self.use_point_shift = use_point_shift

        if self.use_point_stn:
            self.transform_net = Transform_Net()

        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        # self.bn5 = nn.BatchNorm1d(self.emb_dims)  # the emb_dims of each point

        self.conv1 = nn.Sequential(nn.Conv2d(c_in*2, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=True),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=True),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=True),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        # for p in self.parameters():
        #     p.requires_grad = False

        if self.use_point_shift:
            self.shift_net = Point_Shitf_Net()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].transpose(1, 2).contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        """
        :param pointcloud: B*3*1024
        :return:
        """
        # maybe use fixed graph is better for normal estiamtion task ??
        xyz, features = self._break_up_pc(pointcloud) # xyz:[B,3,points_num]
        batch_size = xyz.size(0)
        xyz_t = xyz
        # layer0: QTN
        if self.use_point_stn:
            x0 = get_graph_feature(self.opt, xyz_t,k=self.k)
            trans = self.transform_net(x0)
            xyz = xyz.transpose(2,1)
            xyz = torch.bmm(xyz,trans)
            xyz = xyz.transpose(2,1)
        else:
            trans = None
        if self.use_point_shift:
            x0_1 = get_graph_feature(self.opt, xyz, k=self.k)
            offsets = self.shift_net(x0_1)
            # offsets = 0.2*torch.tanh(offsets) #hope to obtain a small offset ??
            # xyz = xyz + offsets
        else:
            offsets = None


        # layer 1: c_in(3d) ==> 64d
        x1 = get_graph_feature(self.opt,xyz, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # layer 2: 64d ==> 64d
        x2 = get_graph_feature(self.opt,x1, k=self.k, d=4)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        x2 = x2+x1


        # layer 3: 64d ==> 64d
        x3 = get_graph_feature(self.opt,x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        x3 = x3+x2

        # layer 4: 64d ==>64d
        x4 = get_graph_feature(self.opt,x3, k=self.k,  d=4)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        x4 = x4+x3


        return x2, x3, x4, trans, offsets, xyz


class PatchNet(nn.Module):
    def __init__(self,opt, num_points=1024,use_point_stn=True,use_point_shift=True,is_multi_scale=False,add_atten=False):
        super(PatchNet,self).__init__()
        self.is_multi_scale = is_multi_scale
        self.use_point_shift = use_point_shift
        self.add_atten = add_atten
        self.num_points = num_points
        self.opt = opt
        self.feat_out = [64,64,64,64]
        self.feat = PointFeatureNet(opt = opt,
                                    input_channels=3,
                                    use_point_stn=use_point_stn,
                                    use_point_shift = use_point_shift)
        # layer1:
        self.fc_layer1_1 = nn.Sequential(
            nn.Conv1d(self.feat_out[1]+self.feat_out[2]+self.feat_out[3],64,kernel_size=1,bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1_1 = nn.Dropout(p=0.3)
        self.fc_layer1_2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size=1,bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1_2 = nn.Dropout(p=0.3)
        self.fc_layer1_3 = nn.Sequential(nn.Conv1d(32,3,kernel_size=1,bias=True))

        # layer2:
        self.fc_layer2_1 = nn.Sequential(
            nn.Conv1d(self.feat_out[1]+self.feat_out[2]+self.feat_out[3],64,kernel_size=1,bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp2_1 = nn.Dropout(p=0.3)
        self.fc_layer2_2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size=1,bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp2_2 = nn.Dropout(p=0.3)
        self.fc_layer2_3 =nn.Sequential(nn.Conv1d(32,3,kernel_size=1,bias=True))

        # layer3:
        self.fc_layer3_1 = nn.Sequential(
            nn.Conv1d(self.feat_out[1]+self.feat_out[2]+self.feat_out[3],64,kernel_size=1,bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp3_1 = nn.Dropout(p=0.3)
        self.fc_layer3_2 = nn.Sequential(
            nn.Conv1d(64,32,kernel_size=1,bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp3_2 = nn.Dropout(p=0.3)
        self.fc_layer3_3 =nn.Sequential(nn.Conv1d(32,3,kernel_size=1,bias=True))

        if self.is_multi_scale == True:
            self.multi_scale_net = ExpertsNet()


        if self.add_atten==True:
            encoder_layer = TransformerEncoderLayer(d_model=self.feat_out[1]+self.feat_out[2]+self.feat_out[3], nhead=8, dim_feedforward=2048, dropout=0.0)
            self.attn = TransformerEncoder(encoder_layer, num_layers=1)
            # the temperature is just a scalar learnable that controls the softmax strength
            self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self,pointcloud: torch.cuda.FloatTensor, r:torch.cuda.FloatTensor):
        [x2_0,x3_0,x4_0, trans,offsets,xyz] = self.feat(pointcloud)
        f0 = torch.cat([x2_0, x3_0, x4_0], 1)
        f0_x = f0.unsqueeze(dim=-1)

        if self.add_atten==True:
            f0_x = f0_x / self.temp
            f0, weights = run_attn(attn=self.attn, x=f0_x.permute(0, 2, 3, 1), use_ffn=True)
            f0 = f0.squeeze().permute(0,2,1)
            weights = weights.squeeze()
        else:
            weights = None

        # layer1: (use from the second layer features of the dgcnn)
        if self.use_point_shift==True:
            batch_size = xyz.size(0)
            num_points = xyz.size(2)
            xyz_move = xyz+r*offsets  #r=0 or 1
            idx = knn2(xyz_move,xyz,k=1)  # (batch_size, num_points, k)

            # device = torch.device('cuda')
            device = torch.device("cpu" if self.opt.gpu_idx < 0 else "cuda:%d" % self.opt.gpu_idx)
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)




        x2 = self.fc_layer1_1(f0)
        x2 = self.dp1_1(x2)
        x2 = self.fc_layer1_2(x2)
        x2 = self.dp1_2(x2)
        x2 = self.fc_layer1_3(x2)

        # layer2:
        x3 = self.fc_layer2_1(f0)
        x3 = self.dp2_1(x3)
        x3 = self.fc_layer2_2(x3)
        x3 = self.dp2_2(x3)
        x3 = self.fc_layer2_3(x3)

        # layer3:
        x4 = self.fc_layer3_1(f0)
        x4 = self.dp3_1(x4)
        x4 = self.fc_layer3_2(x4)
        x4 = self.dp3_2(x4)
        x4 = self.fc_layer3_3(x4)

        if self.use_point_shift == True:
            x2_t = x2.transpose(2, 1).contiguous()
            x3_t = x3.transpose(2, 1).contiguous()
            x4_t = x4.transpose(2, 1).contiguous()

            x2_t = x2_t.view(batch_size * num_points, -1)
            x3_t = x3_t.view(batch_size * num_points, -1)
            x4_t= x4_t.view(batch_size * num_points, -1)

            x2_t = x2_t[idx,:]
            x3_t = x3_t[idx,:]
            x4_t = x4_t[idx,:]

            x2_t = x2_t.view(batch_size, num_points, 3)
            x3_t = x3_t.view(batch_size, num_points, 3)
            x4_t = x4_t.view(batch_size, num_points, 3)

            x2_t = x2_t.transpose(2, 1).contiguous()
            x3_t = x3_t.transpose(2, 1).contiguous()
            x4_t = x4_t.transpose(2, 1).contiguous()

        if self.is_multi_scale:
            # f = torch.cat([x2_0,x3_0,x4_0],1)
            scales_v = self.multi_scale_net(f0)
        else:
            scales_v = None
        return x2_t, x3_t, x4_t, trans, offsets, scales_v, weights









