# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# 
# This code is from: https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from network.utils import BNReLU


class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channel)
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = self.linear(x).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        return F.relu(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class SpatialGather_Module_ICAW(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.

        Output:
          The correlation of every class map with every feature map
          shape = [n, num_feats, num_classes, 1]


    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module_trans, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), \
            probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)#batch x k x hw
        feats = feats.view(batch_size, feats.size(1), -1)#batch x c x hw

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        prob = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(prob, feats) #batch*k*c矩阵乘法
        return probs, ocr_context

class ObjectAttentionBlock_ICAW(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(ObjectAttentionBlock_trans, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialOCR_Module_ICAW(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.1):
        super(SpatialOCR_Module_trans, self).__init__()
        self.compress = nn.Linear(in_channels, key_channels)
        self.norm1 = nn.LayerNorm(key_channels)
        self.object_context_block = ObjectAttentionBlock_trans(key_channels, num_heads=8)
        if cfg.MODEL.OCR_ASPP:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=cfg.MODEL.ASPP_BOT_CH,
                output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.norm2 = nn.LayerNorm(key_channels)
        self.expand = nn.Linear(key_channels, in_channels)

        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(in_channels),
        )

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, probs, proxy_feats):
        proxy_feats = self.compress(proxy_feats)
        proxy_feats = self.norm1(proxy_feats)

        context = self.object_context_block(proxy_feats)
        context = self.norm2(context)
        context = self.expand(context)

        context = torch.matmul(context.transpose(-2, -1), probs)
        context = context.view(*feats.size())

        context = self.f_up(context)

        if cfg.MODEL.OCR_ASPP:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output