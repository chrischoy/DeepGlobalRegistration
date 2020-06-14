# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import sys
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy
import MinkowskiEngine as ME

sys.path.append('.')
from model import load_model

from core.registration import GlobalRegistration
from core.knn import find_knn_gpu

from util.timer import Timer
from util.pointcloud import make_open3d_point_cloud


# Feature-based registrations in Open3D
def registration_ransac_based_on_feature_matching(pcd0, pcd1, feats0, feats1,
                                                  distance_threshold, num_iterations):
  assert feats0.shape[1] == feats1.shape[1]

  source_feat = o3d.registration.Feature()
  source_feat.resize(feats0.shape[1], len(feats0))
  source_feat.data = feats0.astype('d').transpose()

  target_feat = o3d.registration.Feature()
  target_feat.resize(feats1.shape[1], len(feats1))
  target_feat.data = feats1.astype('d').transpose()

  result = o3d.registration.registration_ransac_based_on_feature_matching(
      pcd0, pcd1, source_feat, target_feat, distance_threshold,
      o3d.registration.TransformationEstimationPointToPoint(False), 4,
      [o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
      o3d.registration.RANSACConvergenceCriteria(num_iterations, 1000))

  return result.transformation


def registration_ransac_based_on_correspondence(pcd0, pcd1, idx0, idx1,
                                                distance_threshold, num_iterations):
  corres = np.stack((idx0, idx1), axis=1)
  corres = o3d.utility.Vector2iVector(corres)

  result = o3d.registration.registration_ransac_based_on_correspondence(
      pcd0, pcd1, corres, distance_threshold,
      o3d.registration.TransformationEstimationPointToPoint(False), 4,
      o3d.registration.RANSACConvergenceCriteria(4000000, num_iterations))

  return result.transformation


class DeepGlobalRegistration:
  def __init__(self, config, device=torch.device('cuda')):
    # Basic config
    self.config = config
    self.clip_weight_thresh = self.config.clip_weight_thresh
    self.device = device

    # Safeguard
    self.safeguard_method = 'correspondence'  # correspondence, feature_matching

    # Final tuning
    self.use_icp = True

    # Misc
    self.feat_timer = Timer()
    self.reg_timer = Timer()

    # Model config loading
    print("=> loading checkpoint '{}'".format(config.weights))
    assert os.path.exists(config.weights)

    state = torch.load(config.weights)
    network_config = state['config']
    self.network_config = network_config
    self.config.inlier_feature_type = network_config.inlier_feature_type
    self.voxel_size = network_config.voxel_size
    print(f'=> Setting voxel size to {self.voxel_size}')

    # FCGF network initialization
    num_feats = 1
    try:
      FCGFModel = load_model(network_config['feat_model'])
      self.fcgf_model = FCGFModel(
          num_feats,
          network_config['feat_model_n_out'],
          bn_momentum=network_config['bn_momentum'],
          conv1_kernel_size=network_config['feat_conv1_kernel_size'],
          normalize_feature=network_config['normalize_feature'])

    except KeyError:  # legacy pretrained models
      FCGFModel = load_model(network_config['model'])
      self.fcgf_model = FCGFModel(num_feats,
                                  network_config['model_n_out'],
                                  bn_momentum=network_config['bn_momentum'],
                                  conv1_kernel_size=network_config['conv1_kernel_size'],
                                  normalize_feature=network_config['normalize_feature'])

    self.fcgf_model.load_state_dict(state['state_dict'])
    self.fcgf_model = self.fcgf_model.to(device)
    self.fcgf_model.eval()

    # Inlier network initialization
    num_feats = 6 if network_config.inlier_feature_type == 'coords' else 1
    InlierModel = load_model(network_config['inlier_model'])
    self.inlier_model = InlierModel(
        num_feats,
        1,
        bn_momentum=network_config['bn_momentum'],
        conv1_kernel_size=network_config['inlier_conv1_kernel_size'],
        normalize_feature=False,
        D=6)

    self.inlier_model.load_state_dict(state['state_dict_inlier'])
    self.inlier_model = self.inlier_model.to(self.device)
    self.inlier_model.eval()
    print("=> loading finished")

  def preprocess(self, pcd):
    '''
    Stage 0: preprocess raw input point cloud
    Input: raw point cloud
    Output: voxelized point cloud with
    - xyz:    unique point cloud with one point per voxel
    - coords: coords after voxelization
    - feats:  dummy feature placeholder for general sparse convolution
    '''
    if isinstance(pcd, o3d.geometry.PointCloud):
      xyz = np.array(pcd.points)
    elif isinstance(pcd, np.ndarray):
      xyz = pcd
    else:
      raise Exception('Unrecognized pcd type')

    # Voxelization:
    # Maintain double type for xyz to improve numerical accuracy in quantization
    sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
    npts = len(sel)

    xyz = torch.from_numpy(xyz[sel])

    # ME standard batch coordinates
    coords = ME.utils.batched_coordinates([torch.floor(xyz / self.voxel_size).int()])
    feats = torch.ones(npts, 1)

    return xyz.float(), coords, feats

  def fcgf_feature_extraction(self, feats, coords):
    '''
    Step 1: extract fast and accurate FCGF feature per point
    '''
    sinput = ME.SparseTensor(feats, coords=coords).to(self.device)

    return self.fcgf_model(sinput).F

  def fcgf_feature_matching(self, feats0, feats1):
    '''
    Step 2: coarsely match FCGF features to generate initial correspondences
    '''
    nns = find_knn_gpu(feats0,
                       feats1,
                       nn_max_n=self.network_config.nn_max_n,
                       knn=1,
                       return_distance=False)
    corres_idx0 = torch.arange(len(nns)).long().squeeze()
    corres_idx1 = nns.long().squeeze()

    return corres_idx0, corres_idx1

  def inlier_feature_generation(self, xyz0, xyz1, coords0, coords1, fcgf_feats0,
                                fcgf_feats1, corres_idx0, corres_idx1):
    '''
    Step 3: generate features for inlier prediction
    '''
    assert len(corres_idx0) == len(corres_idx1)

    feat_type = self.config.inlier_feature_type
    assert feat_type in ['ones', 'feats', 'coords']

    corres_idx0 = corres_idx0.to(self.device)
    corres_idx1 = corres_idx1.to(self.device)

    if feat_type == 'ones':
      feat = torch.ones((len(corres_idx0), 1)).float()
    elif feat_type == 'feats':
      feat = torch.cat((fcgf_feats0[corres_idx0], fcgf_feats1[corres_idx1]), dim=1)
    elif feat_type == 'coords':
      feat = torch.cat((torch.cos(xyz0[corres_idx0]), torch.cos(xyz1[corres_idx1])),
                       dim=1)
    else:  # should never reach here
      raise TypeError('Undefined feature type')

    return feat

  def inlier_prediction(self, inlier_feats, coords):
    '''
    Step 4: predict inlier likelihood
    '''
    sinput = ME.SparseTensor(inlier_feats, coords=coords).to(self.device)
    soutput = self.inlier_model(sinput)

    return soutput.F

  def safeguard_registration(self, pcd0, pcd1, idx0, idx1, feats0, feats1,
                             distance_threshold, num_iterations):
    if self.safeguard_method == 'correspondence':
      T = registration_ransac_based_on_correspondence(pcd0,
                                                      pcd1,
                                                      idx0.cpu().numpy(),
                                                      idx1.cpu().numpy(),
                                                      distance_threshold,
                                                      num_iterations=num_iterations)
    elif self.safeguard_method == 'fcgf_feature_matching':
      T = registration_ransac_based_on_fcgf_feature_matching(pcd0, pcd1,
                                                             feats0.cpu().numpy(),
                                                             feats1.cpu().numpy(),
                                                             distance_threshold,
                                                             num_iterations)
    else:
      raise ValueError('Undefined')
    return T

  def register(self, xyz0, xyz1, inlier_thr=0.00):
    '''
    Main algorithm of DeepGlobalRegistration
    '''
    self.reg_timer.tic()
    with torch.no_grad():
      # Step 0: voxelize and generate sparse input
      xyz0, coords0, feats0 = self.preprocess(xyz0)
      xyz1, coords1, feats1 = self.preprocess(xyz1)

      # Step 1: Feature extraction
      self.feat_timer.tic()
      fcgf_feats0 = self.fcgf_feature_extraction(feats0, coords0)
      fcgf_feats1 = self.fcgf_feature_extraction(feats1, coords1)
      self.feat_timer.toc()

      # Step 2: Coarse correspondences
      corres_idx0, corres_idx1 = self.fcgf_feature_matching(fcgf_feats0, fcgf_feats1)

      # Step 3: Inlier feature generation
      # coords[corres_idx0]: 1D temporal + 3D spatial coord
      # coords[corres_idx1, 1:]: 3D spatial coord
      # => 1D temporal + 6D spatial coord
      inlier_coords = torch.cat((coords0[corres_idx0], coords1[corres_idx1, 1:]),
                                dim=1).int()
      inlier_feats = self.inlier_feature_generation(xyz0, xyz1, coords0, coords1,
                                                    fcgf_feats0, fcgf_feats1,
                                                    corres_idx0, corres_idx1)

      # Step 4: Inlier likelihood estimation and truncation
      logit = self.inlier_prediction(inlier_feats.contiguous(), coords=inlier_coords)
      weights = logit.sigmoid()
      if self.clip_weight_thresh > 0:
        weights[weights < self.clip_weight_thresh] = 0
      wsum = weights.sum().item()

    # Step 5: Registration. Note: torch's gradient may be required at this stage
    # > Case 0: Weighted Procrustes + Robust Refinement
    wsum_threshold = max(200, len(weights) * 0.05)
    sign = '>=' if wsum >= wsum_threshold else '<'
    print(f'=> Weighted sum {wsum:.2f} {sign} threshold {wsum_threshold}')

    T = np.identity(4)
    if wsum >= wsum_threshold:
      try:
        rot, trans, opt_output = GlobalRegistration(xyz0[corres_idx0],
                                                    xyz1[corres_idx1],
                                                    weights=weights.detach().cpu(),
                                                    break_threshold_ratio=1e-4,
                                                    quantization_size=2 *
                                                    self.voxel_size,
                                                    verbose=False)
        T[0:3, 0:3] = rot.detach().cpu().numpy()
        T[0:3, 3] = trans.detach().cpu().numpy()
        dgr_time = self.reg_timer.toc()
        print(f'=> DGR takes {dgr_time:.2} s')

      except RuntimeError:
        # Will directly go to Safeguard
        print('###############################################')
        print('# WARNING: SVD failed, weights sum: ', wsum)
        print('# Falling back to Safeguard')
        print('###############################################')

    else:
      # > Case 1: Safeguard RANSAC + (Optional) ICP
      pcd0 = make_open3d_point_cloud(xyz0)
      pcd1 = make_open3d_point_cloud(xyz1)
      T = self.safeguard_registration(pcd0,
                                      pcd1,
                                      corres_idx0,
                                      corres_idx1,
                                      feats0,
                                      feats1,
                                      2 * self.voxel_size,
                                      num_iterations=80000)
      safeguard_time = self.reg_timer.toc()
      print(f'=> Safeguard takes {safeguard_time:.2} s')

    if self.use_icp:
      T = o3d.registration.registration_icp(
          make_open3d_point_cloud(xyz0),
          make_open3d_point_cloud(xyz1), self.voxel_size * 2, T,
          o3d.registration.TransformationEstimationPointToPoint()).transformation

    return T
