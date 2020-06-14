# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch.nn as nn
import MinkowskiEngine as ME


def get_norm(norm_type, num_feats, bn_momentum=0.05, dimension=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats)
  elif norm_type == 'INBN':
    return nn.Sequential(
        ME.MinkowskiInstanceNorm(num_feats),
        ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum))
  else:
    raise ValueError(f'Type {norm_type}, not defined')


def get_nonlinearity(non_type):
  if non_type == 'ReLU':
    return ME.MinkowskiReLU()
  elif non_type == 'ELU':
    # return ME.MinkowskiInstanceNorm(num_feats, dimension=dimension)
    return ME.MinkowskiELU()
  else:
    raise ValueError(f'Type {non_type}, not defined')
