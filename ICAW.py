# !/usr/bin/env Python
# coding=utf-8
"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from torch import nn

from network.mynn import initialize_weights, Upsample, scale_as
from network.mynn import ResizeX
from network.utils import get_trunk
from network.utils import BNReLU
from network.utils import make_attn_head
from network.ocr_utils import SpatialGather_Module_ICAW, SpatialOCR_Module_ICAW
from config import cfg
from utils.misc import fmt_scale


class ICAW_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, high_level_ch):
        super(OCR_block, self).__init__()

        ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module_ICAW(num_classes)
        self.ocr_distri_head = SpatialOCR_Module_ICAW(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 dropout=0.05,
                                                 )

        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        probs, context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, probs, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class ICAW(nn.Module):
    """
    OCR net for ICAW
    """
    def __init__(self, num_classes, trunk='resnet50', criterion=None):
        super(OCRNet, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = ICAW_block(high_level_ch)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)

        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            aux_loss = self.criterion(aux_out, gts,
                                      do_rmi=cfg.LOSS.OCR_AUX_RMI)
            main_loss = self.criterion(cls_out, gts)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict

def HRNet(num_classes, criterion):
    return OCRNet(num_classes, trunk='hrnetv2', criterion=criterion)

def SETR(num_classes, criterion):
    return OCRNet(num_classes, trunk='setr', criterion=criterion)

def SegFormer(num_classes, criterion):
    return OCRNet(num_classes, trunk='segformer', criterion=criterion)

def ResNet(num_classes, criterion):
    return OCRNet(num_classes, trunk='resnet50', criterion=criterion)

def Xception(num_classes, criterion):
    return OCRNet(num_classes, trunk='xception', criterion=criterion)

def Mbv3(num_classes, criterion):
    return OCRNet(num_classes, trunk='mobilenetv3', criterion=criterion)

def GoogleNet(num_classes, criterion):
    return OCRNet(num_classes, trunk='googlenet', criterion=criterion)
