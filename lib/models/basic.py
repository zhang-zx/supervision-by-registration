# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .cpm_vgg16 import cpm_vgg16
from .cpm_vgg11 import cpm_vgg11
from .LK import LK
from .cpm_small import cpm_small
from .cpm_mobileNet import cpm_mobile

def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'cpm_vgg11':
    net = cpm_vgg11(configure, points)
  elif configure.arch == 'cpm_small':
    net = cpm_small(configure, points)
  elif configure.arch == 'mobile':
    net = cpm_mobile(configure, points)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net

def obtain_LK(configure, lkconfig, points):
  model = obtain_model(configure, points)
  lk_model = LK(model, lkconfig, points)
  return lk_model
