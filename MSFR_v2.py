import torch.nn as nn
import network.mynn as mynn
import torch.nn.functional as F
from network.mynn import Norm2d as BatchNorm2d
from network.mynn import initialize_weights, Upsample, Norm2d
import torch
from config import cfg
import os
from runx.logx import logx
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class MSEA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSEA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x2 + x2 * y.expand_as(x2)

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x2 + x1 * y.expand_as(x2)

class ResNet_MSFR_v2(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000, criterion=None):
        self.inplanes = 64
        super(MFR_ResNet, self).__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = mynn.Norm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.PSA_1 = self.make_attn(64, 1)
        self.SEA_2 = MSEA(128)
        self.SEA_3 = MSEA(256)
        self.SEA_4 = MSEA(256)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_4 = ConvBNReLU(2048, 256, ks=3, padding=1)

        self.conv_3 = ConvBNReLU(1024, 256, ks=3, padding=1)
        self.conv_2 = ConvBNReLU(512, 128, ks=3, padding=1)

        self.conv_fuse0 = ConvBNReLU(256, 256, ks=3, padding=1)
        self.conv_fuse1 = ConvBNReLU(256, 128, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(128, 256, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(256, 64, ks=3, padding=1)

        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)

        self.conv_out = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)

    def make_attn(self, in_ch, out_ch):

        attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      padding=1, bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid())

        return attention

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']
        x_size = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.conv_4(x4)

        feat16_low = self.conv_3(x3)
        feat8_low = self.conv_2(x2)

        PSA4_low = self.PSA_1(x)

        H, W = x3.size()[2:]
        feat_lkpp_up = F.interpolate(self.conv_fuse0(self.SEA_4(x4, x4)), (H, W), mode='bilinear',
                                     align_corners=True)

        feat_out = self.conv_fuse1(self.SEA_3(feat16_low, feat_lkpp_up))

        H, W = x2.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)

        feat_out = self.conv_fuse2(self.SEA_2(feat8_low, feat_out))

        H, W = x.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)
        feat = F.interpolate(x4, (H, W), mode='bilinear',
                                     align_corners=True)

        feat_out = self.conv_fuse3(feat_out + feat * PSA4_low)

        final = self.conv_out(self.fuse(feat_out))

        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)

        return {'pred': out}


    def init_weights(self, pretrained=cfg.MODEL.RESNET_CHECKPOINT):
        logx.msg('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, cfg.MODEL.BNFUNC):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location={'cuda:0': 'cpu'})
            logx.msg('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer',
                                         'aux_head').replace('model.', ''): v
                               for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

class FWD_ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000, criterion=None):
        self.inplanes = 64
        super(FWD_ResNet, self).__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = mynn.Norm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.PSA_1 = self.make_attn(64, 1)
        self.SEA_2 = SE(128)
        self.SEA_3 = SE(256)
        self.SEA_4 = SE(256)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_4 = ConvBNReLU(2048, 256, ks=3, padding=1)
        self.conv_3 = ConvBNReLU(1024, 256, ks=3, padding=1)
        self.conv_2 = ConvBNReLU(512, 128, ks=3, padding=1)
        self.conv_1 = ConvBNReLU(256, 64, ks=3, padding=1)

        self.conv_fuse0 = ConvBNReLU(256, 256, ks=3, padding=1)
        self.conv_fuse1 = ConvBNReLU(256, 128, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(128, 64, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(64, 64, ks=3, padding=1)

        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)

        self.conv_out = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)

    def make_attn(self, in_ch, out_ch):

        attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      padding=1, bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid())

        return attention

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']
        x_size = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.conv_4(x4)

        feat16_low = self.conv_3(x3)
        feat8_low = self.conv_2(x2)
        feat4_low = self.conv_1(x1)

        H, W = x3.size()[2:]
        feat_lkpp_up = F.interpolate(self.conv_fuse0(self.SEA_4(x4, x4)), (H, W), mode='bilinear',
                                     align_corners=True)

        feat_out = self.conv_fuse1(self.SEA_3(feat16_low, feat_lkpp_up))

        H, W = x2.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)

        feat_out = self.conv_fuse2(self.SEA_2(feat8_low, feat_out))

        H, W = x.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)

        feat_out = self.conv_fuse3(feat_out + feat4_low * self.PSA_1(feat4_low))

        final = self.conv_out(self.fuse(feat_out))

        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)

        return {'pred': out}


    def init_weights(self, pretrained=cfg.MODEL.RESNET_CHECKPOINT):
        logx.msg('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, cfg.MODEL.BNFUNC):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location={'cuda:0': 'cpu'})
            logx.msg('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer',
                                         'aux_head').replace('model.', ''): v
                               for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

class FW2D_ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000, criterion=None):
        self.inplanes = 64
        super(FW2D_ResNet, self).__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = mynn.Norm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.PSA_1 = self.make_attn(64, 1)
        self.SEA_2 = SE(128)
        self.SEA_3 = SE(256)
        self.SEA_4 = SE(256)

        # self.SWD_1 = self.make_attn(256, 1)
        # self.SWD_2 = self.make_attn(512, 1)
        # self.SWD_3 = self.make_attn(1024, 1)
        self.CWD_1 = SE(256)
        self.CWD_2 = SE(512)
        self.CWD_3 = SE(1024)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_4 = ConvBNReLU(2048, 256, ks=3, padding=1)
        self.conv_3 = ConvBNReLU(1024, 256, ks=3, padding=1)
        self.conv_2 = ConvBNReLU(512, 128, ks=3, padding=1)
        self.conv_1 = ConvBNReLU(256, 64, ks=3, padding=1)

        self.conv_fuse0 = ConvBNReLU(256, 256, ks=3, padding=1)
        self.conv_fuse1 = ConvBNReLU(256, 128, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(128, 64, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(64, 64, ks=3, padding=1)

        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)

        self.conv_out = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)

    def make_attn(self, in_ch, out_ch):

        attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      padding=1, bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False),
            Norm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid())

        return attention

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(self.CWD_1(x1, x1))
        x3 = self.layer3(self.CWD_2(x2, x2))
        x4 = self.layer4(self.CWD_3(x3, x3))

        x4 = self.conv_4(x4)

        feat16_low = self.conv_3(x3)
        feat8_low = self.conv_2(x2)
        feat4_low = self.conv_1(x1)

        H, W = x3.size()[2:]
        feat_lkpp_up = F.interpolate(self.conv_fuse0(self.SEA_4(x4, x4)), (H, W), mode='bilinear',
                                     align_corners=True)

        feat_out = self.conv_fuse1(self.SEA_3(feat16_low, feat_lkpp_up))

        H, W = x2.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)

        feat_out = self.conv_fuse2(self.SEA_2(feat8_low, feat_out))

        H, W = x.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)

        feat_out = self.conv_fuse3(feat_out + feat4_low * self.PSA_1(feat4_low))

        return None, None, feat_out


    def init_weights(self, pretrained=cfg.MODEL.RESNET_CHECKPOINT):
        logx.msg('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, cfg.MODEL.BNFUNC):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location={'cuda:0': 'cpu'})
            logx.msg('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer',
                                         'aux_head').replace('model.', ''): v
                               for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

def get_seg_model():
    model = FW2D_ResNet(Bottleneck, [3, 4, 6, 3])
    model.init_weights()

    return model

def ResNet50_MSFR_v2(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_MSFR_v2(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.init_weights()
    if pretrained:
        pretrained_dict = torch.load(pretrained,
                                     map_location={'cuda:0': 'cpu'})
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def MSFR_v2(num_classes, trunk='resnet50', criterion=None):
    return ResNet50_MSFR_v2(pretrained=cfg.MODEL.RESNET50_CHECKPOINT, num_classes=num_classes, criterion=criterion)

def FWD_ResNet50(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FWD_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.init_weights()
    if pretrained:
        pretrained_dict = torch.load(pretrained,
                                     map_location={'cuda:0': 'cpu'})
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def FWD(num_classes, trunk='resnet50', criterion=None):
    return MFR_ResNet50(pretrained=cfg.MODEL.RESNET50_CHECKPOINT, num_classes=num_classes, criterion=criterion)