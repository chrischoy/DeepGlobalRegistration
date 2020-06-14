# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import logging
import random
import torch
import torch.utils.data
import numpy as np

import dataloader.transforms as t
from dataloader.inf_sampler import InfSampler

import MinkowskiEngine as ME
import open3d as o3d


class CollationFunctionFactory:
  def __init__(self, concat_correspondences=True, collation_type='default'):
    self.concat_correspondences = concat_correspondences
    if collation_type == 'default':
      self.collation_fn = self.collate_default
    elif collation_type == 'collate_pair':
      self.collation_fn = self.collate_pair_fn
    else:
      raise ValueError(f'collation_type {collation_type} not found')

  def __call__(self, list_data):
    return self.collation_fn(list_data)

  def collate_default(self, list_data):
    return list_data

  def collate_pair_fn(self, list_data):
    N = len(list_data)
    list_data = [data for data in list_data if data is not None]
    if N != len(list_data):
      logging.info(f"Retain {len(list_data)} from {N} data.")
    if len(list_data) == 0:
      raise ValueError('No data in the batch')

    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, extra_packages = list(
        zip(*list_data))
    matching_inds_batch, trans_batch, len_batch = [], [], []

    coords_batch0 = ME.utils.batched_coordinates(coords0)
    coords_batch1 = ME.utils.batched_coordinates(coords1)
    trans_batch = torch.from_numpy(np.stack(trans)).float()

    curr_start_inds = torch.zeros((1, 2), dtype=torch.int32)
    for batch_id, _ in enumerate(coords0):
      # For scan2cad there will be empty matching_inds even after filtering
      # This check will skip these pairs while not affecting other datasets
      if (len(matching_inds[batch_id]) == 0):
        continue

      N0 = coords0[batch_id].shape[0]
      N1 = coords1[batch_id].shape[0]

      if self.concat_correspondences:
        matching_inds_batch.append(
            torch.IntTensor(matching_inds[batch_id]) + curr_start_inds)
      else:
        matching_inds_batch.append(torch.IntTensor(matching_inds[batch_id]))

      len_batch.append([N0, N1])

      # Move the head
      curr_start_inds[0, 0] += N0
      curr_start_inds[0, 1] += N1

    # Concatenate all lists
    feats_batch0 = torch.cat(feats0, 0).float()
    feats_batch1 = torch.cat(feats1, 0).float()
    # xyz_batch0 = torch.cat(xyz0, 0).float()
    # xyz_batch1 = torch.cat(xyz1, 0).float()
    # trans_batch = torch.cat(trans_batch, 0).float()
    if self.concat_correspondences:
      matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        'pcd0': xyz0,
        'pcd1': xyz1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0,
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1,
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch,
        'extra_packages': extra_packages,
    }


class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)
