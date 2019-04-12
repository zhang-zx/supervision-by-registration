# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import os, sys, time, random, argparse, PIL
from os import path as osp
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from xvision import draw_image_by_points
from xvision import Eval_Meta

def visualize(args):
  print ('The result file is {:}'.format(args.meta))
  print ('The save path is {:}'.format(args.save))
  meta1 = Path(args.meta + 'eval-start-eval-00-01.pth')
  meta2 = Path(args.meta + args.model)
  save = Path(args.save)
  assert meta1.exists(), 'The model path {:} does not exist'
  assert meta2.exists(), 'The model path {:} does not exist'
  xmeta1 = Eval_Meta()
  xmeta1.load(meta1)
  xmeta2 = Eval_Meta()
  xmeta2.load(meta2)
  assert len(xmeta1)==len(xmeta2), 'different length to compare'
  print ('this meta file has {:} predictions'.format(len(xmeta1)))
  if not save.exists(): os.makedirs( args.save )
  for i in range(len(xmeta1)):
    image1, prediction1 = xmeta1.image_lists[i], xmeta1.predictions[i]
    name = osp.basename(image1)
    image1 = draw_image_by_points(image1, prediction1, 2, (255, 0, 0), False, False)
    image2, prediction2 = xmeta2.image_lists[i], xmeta2.predictions[i]
    name = osp.basename(image2)
    image2 = draw_image_by_points(image2, prediction2, 2, (255, 0, 0), False, False)
    path = save / name
    images = [image1, image2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      image.paste(im, (x_offset, 0))
      x_offset += im.size[0]
    image.save(path)
    print ('{:03d}-th image is saved into {:}'.format(i, path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='visualize the results on a single ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--meta',            type=str,   help='The evaluation image path.')
  parser.add_argument('--model',           type=str,   help='model to compare')
  parser.add_argument('--save',            type=str,   help='The path to save the visualized results.')
  args = parser.parse_args()
  visualize(args)
