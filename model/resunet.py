# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm

from model.residual_block import conv, conv_tr, get_block


class ResUNet(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128]
  TR_CHANNELS = [None, 32, 64, 64]
  REGION_TYPE = ME.RegionType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature

    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[1],
        CHANNELS[1],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[2],
        CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[3],
        CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[3],
        TR_CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[2],
        TR_CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNetBN(ResUNet):
  NORM_TYPE = 'BN'


class ResUNetBNF(ResUNet):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64]
  TR_CHANNELS = [None, 16, 32, 64]


class ResUNetBNFX(ResUNetBNF):
  REGION_TYPE = ME.RegionType.HYPER_CROSS


class ResUNetSP(ME.MinkowskiNetwork):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128]
  TR_CHANNELS = [None, 32, 64, 64]
  # None        b1, b2, b3, btr3, btr2
  #               1  2  3 -3 -2 -1
  DEPTHS = [None, 1, 1, 1, 1, 1, None]
  REGION_TYPE = ME.RegionType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[1])
    ])

    self.pool2 = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[2])
    ])

    self.pool3 = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[3])
    ])

    self.pool3_tr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D)
    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3],
        out_channels=TR_CHANNELS[3],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-3])
    ])

    self.pool2_tr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D)
    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-2])
    ])

    self.conv1_tr = conv_tr(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.final = conv(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    out_s2 = self.pool2(out_s1)
    out_s2 = self.conv2(out_s2)
    out_s2 = self.norm2(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)

    out_s4 = self.pool3(out_s2)
    out_s4 = self.conv3(out_s4)
    out_s4 = self.norm3(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)

    out_s2t = self.pool3_tr(out_s4)
    out_s2t = self.conv3_tr(out_s2t)
    out_s2t = self.norm3_tr(out_s2t)
    out_s2t = MEF.relu(out_s2t)
    out_s2t = self.block3_tr(out_s2t)

    out = ME.cat(out_s2t, out_s2)

    out_s1t = self.conv2_tr(out)
    out_s1t = self.pool3_tr(out_s1t)
    out_s1t = self.norm2_tr(out_s1t)
    out_s1t = MEF.relu(out_s1t)
    out_s1t = self.block2_tr(out_s1t)

    out = ME.cat(out_s1t, out_s1)

    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNetBNSPC(ResUNetSP):
  REGION_TYPE = ME.RegionType.HYPER_CROSS


class ResUNetINBNSPC(ResUNetBNSPC):
  NORM_TYPE = 'INBN'


class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  REGION_TYPE = ME.RegionType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[1],
        CHANNELS[1],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[2],
        CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[3],
        CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv4 = conv(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[4],
        CHANNELS[4],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv4_tr = conv_tr(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[4],
        TR_CHANNELS[4],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[3],
        TR_CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[2],
        TR_CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv1_tr = conv(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2CX(ResUNetBN2C):
  REGION_TYPE = ME.RegionType.HYPER_CROSS


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetBN2F(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]


class ResUNetBN2FX(ResUNetBN2F):
  REGION_TYPE = ME.RegionType.HYPER_CROSS


class ResUNet2v2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  # None        b1, b2, b3, b4, btr4, btr3, btr2
  #               1  2  3  4,-4,-3,-2,-1
  DEPTHS = [None, 1, 1, 1, 1, 1, 1, 1, None]
  REGION_TYPE = ME.RegionType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    self.normalize_feature = normalize_feature

    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[1])
    ])

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[2])
    ])

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[3])
    ])

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[4],
            CHANNELS[4],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[4])
    ])

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[4],
            TR_CHANNELS[4],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[-4])
    ])

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[-3])
    ])

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            dimension=D) for d in range(DEPTHS[-2])
    ])

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)
    self.weight_initialization()

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def forward(self, x):  # Receptive field size
    out_s1 = self.conv1(x)  # 7
    out_s1 = self.norm1(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    out_s2 = self.conv2(out_s1)  # 7 + 2 * 2 = 11
    out_s2 = self.norm2(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)  # 11 + 2 * (2 + 2) = 19

    out_s4 = self.conv3(out_s2)  # 19 + 4 * 2 = 27
    out_s4 = self.norm3(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)  # 27 + 4 * (2 + 2) = 43

    out_s8 = self.conv4(out_s4)  # 43 + 8 * 2 = 59
    out_s8 = self.norm4(out_s8)
    out_s8 = MEF.relu(out_s8)
    out_s8 = self.block4(out_s8)  # 59 + 8 * (2 + 2) = 91

    out = self.conv4_tr(out_s8)  # 91 + 4 * 2 = 99
    out = self.norm4_tr(out)
    out = MEF.relu(out)
    out = self.block4_tr(out)  # 99 + 4 * (2 + 2) = 115

    out = ME.cat(out, out_s4)

    out = self.conv3_tr(out)  # 115 + 2 * 2 = 119
    out = self.norm3_tr(out)
    out = MEF.relu(out)
    out = self.block3_tr(out)  # 119 + 2 * (2 + 2) = 127

    out = ME.cat(out, out_s2)

    out = self.conv2_tr(out)  # 127 + 2 = 129
    out = self.norm2_tr(out)
    out = MEF.relu(out)
    out = self.block2_tr(out)  # 129 + 1 * (2 + 2) = 133

    out = ME.cat(out, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNetBN2v2(ResUNet2v2):
  NORM_TYPE = 'BN'


class ResUNetBN2Bv2(ResUNet2v2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2Cv2(ResUNet2v2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2Dv2(ResUNet2v2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2Ev2(ResUNet2v2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetBN2Fv2(ResUNet2v2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]


class ResUNet2SP(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  REGION_TYPE = ME.RegionType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        region_type=ME.RegionType.HYPER_CUBE,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[1],
        CHANNELS[1],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.pool2 = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[2],
        CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.pool3 = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[3],
        CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.pool4 = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
    self.conv4 = conv(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        region_type=ME.RegionType.HYPER_CUBE,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE,
        CHANNELS[4],
        CHANNELS[4],
        bn_momentum=bn_momentum,
        region_type=ME.RegionType.HYPER_CUBE,
        dimension=D)

    self.conv4_tr = conv_tr(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=ME.RegionType.HYPER_CUBE,
        dimension=D)
    self.norm4_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[4],
        TR_CHANNELS[4],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[3],
        TR_CHANNELS[3],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE,
        TR_CHANNELS[2],
        TR_CHANNELS[2],
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        dimension=D)

    self.conv1_tr = conv(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.pool2(out)
    out_s2 = self.conv2(out_s2)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.pool3(out)
    out_s4 = self.conv3(out_s4)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.pool4(out)
    out_s8 = self.conv4(out_s8)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager)
    else:
      return out


class ResUNetBN2SPC(ResUNet2SP):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2SPCX(ResUNetBN2SPC):
  REGION_TYPE = ME.RegionType.HYPER_CROSS
