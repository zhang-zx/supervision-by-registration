# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from copy import deepcopy
from .model_utils import get_parameters
from .basic_batch import find_tensor_peak_batch
from .initialization import weights_init_cpm
from .initialization import weights_init_kaiming


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, config, pts_num, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.config = deepcopy(config)
        self.downsample = 8
        self.pts_num = pts_num

        block = InvertedResidual
        input_channel = 32
        last_channel = 64
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.CPM_feature = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),  # CPM_1
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))  # CPM_2

        assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config.stages)
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config.stages):
            stagex = nn.Sequential(
                nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [
            {'params': self.features.parameters(), 'lr': base_lr, 'weight_decay': base_weight_decay},
            # {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
            ]
        for stage in self.stages:
            params_dict.append(
                {'params': get_parameters(stage, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True), 'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    # return : cpm-stages, locations
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms, batch_locs, batch_scos = [], [], []

        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(self.config.stages):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]], 1))
            batch_cpms.append(cpm)

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config.argmax,
                                                                 self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

        return batch_cpms, batch_locs, batch_scos


def cpm_mobile(config, pts):
    print('Initialize cpm-vgg16 with configure : {}'.format(config))
    model = MobileNetV2(config, pts)
    model.apply(weights_init_cpm)
    return model
