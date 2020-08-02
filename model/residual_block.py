# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch.nn as nn

from model.common import get_norm, get_nonlinearity

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


def conv(in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         dilation=1,
         bias=False,
         region_type=0,
         dimension=3):
  if not isinstance(region_type, ME.RegionType):
    if region_type == 0:
      region_type = ME.RegionType.HYPER_CUBE
    elif region_type == 1:
      region_type = ME.RegionType.HYPER_CROSS
    else:
      raise ValueError('Unsupported region type')

  kernel_generator = ME.KernelGenerator(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      region_type=region_type,
      dimension=dimension)

  return ME.MinkowskiConvolution(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      kernel_generator=kernel_generator,
      dimension=dimension)


def conv_tr(in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            region_type=ME.RegionType.HYPER_CUBE,
            dimension=-1):
  assert dimension > 0, 'Dimension must be a positive integer'
  kernel_generator = ME.KernelGenerator(
      kernel_size,
      stride,
      dilation,
      is_transpose=True,
      region_type=region_type,
      dimension=dimension)

  kernel_generator = ME.KernelGenerator(
      kernel_size,
      stride,
      dilation,
      is_transpose=True,
      region_type=region_type,
      dimension=dimension)

  return ME.MinkowskiConvolutionTranspose(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      bias=bias,
      kernel_generator=kernel_generator,
      dimension=dimension)


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               region_type=0,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = conv(
        inplanes,
        planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        region_type=region_type,
        dimension=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        region_type=region_type,
        dimension=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.relu(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


class BasicBlockINBN(BasicBlockBase):
  NORM_TYPE = 'INBN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              region_type=0,
              dimension=3):
  if norm_type == 'BN':
    Block = BasicBlockBN
  elif norm_type == 'IN':
    Block = BasicBlockIN
  elif norm_type == 'INBN':
    Block = BasicBlockINBN
  else:
    raise ValueError(f'Type {norm_type}, not defined')

  return Block(inplanes, planes, stride, dilation, downsample, bn_momentum, region_type,
               dimension)


def conv_norm_non(inc,
                  outc,
                  kernel_size,
                  stride,
                  dimension,
                  bn_momentum=0.05,
                  region_type=ME.RegionType.HYPER_CUBE,
                  norm_type='BN',
                  nonlinearity='ELU'):
  return nn.Sequential(
      conv(
          in_channels=inc,
          out_channels=outc,
          kernel_size=kernel_size,
          stride=stride,
          dilation=1,
          bias=False,
          region_type=region_type,
          dimension=dimension),
      get_norm(norm_type, outc, bn_momentum=bn_momentum, dimension=dimension),
      get_nonlinearity(nonlinearity))
