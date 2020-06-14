# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import time
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
from core.knn import find_knn_batch
from core.correspondence import find_correct_correspondence
from core.loss import UnbalancedLoss, BalancedLoss
from core.metrics import batch_rotation_error, batch_translation_error
import core.registration as GlobalRegistration

from util.timer import Timer, AverageMeter
from util.file import ensure_dir

import MinkowskiEngine as ME

eps = np.finfo(float).eps
np2th = torch.from_numpy


class WeightedProcrustesTrainer:
  def __init__(self, config, data_loader, val_data_loader=None):
    # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.
    num_feats = 3 if config.use_xyz_feature else 1

    # Feature model initialization
    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.config = config

    # Training config
    self.max_epoch = config.max_epoch
    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    self.data_loader = data_loader
    self.train_data_loader_iter = self.data_loader.__iter__()

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size

    # Validation config
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq
    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    self.val_data_loader = val_data_loader
    self.test_valid = True if self.val_data_loader is not None else False

    # Logging
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.writer = SummaryWriter(config.out_dir)

    # Model
    FeatModel = load_model(config.feat_model)
    InlierModel = load_model(config.inlier_model)

    num_feats = 6 if self.config.inlier_feature_type == 'coords' else 1
    self.feat_model = FeatModel(num_feats,
                                config.feat_model_n_out,
                                bn_momentum=config.bn_momentum,
                                conv1_kernel_size=config.feat_conv1_kernel_size,
                                normalize_feature=config.normalize_feature).to(
                                    self.device)
    logging.info(self.feat_model)

    self.inlier_model = InlierModel(num_feats,
                                    1,
                                    bn_momentum=config.bn_momentum,
                                    conv1_kernel_size=config.inlier_conv1_kernel_size,
                                    normalize_feature=False,
                                    D=6).to(self.device)
    logging.info(self.inlier_model)

    # Loss and optimizer
    self.clip_weight_thresh = self.config.clip_weight_thresh
    if self.config.use_balanced_loss:
      self.crit = BalancedLoss()
    else:
      self.crit = UnbalancedLoss()

    self.optimizer = getattr(optim, config.optimizer)(self.inlier_model.parameters(),
                                                      lr=config.lr,
                                                      momentum=config.momentum,
                                                      weight_decay=config.weight_decay)
    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    # Output preparation
    ensure_dir(self.checkpoint_dir)
    json.dump(config,
              open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
              indent=4,
              sort_keys=False)

    self._load_weights(config)

  def train(self):
    """
    Major interface
    Full training logic: train, valid, and save
    """
    # Baseline random feature performance
    if self.test_valid:
      val_dict = self._valid_epoch()
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    # Train and valid
    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        val_dict = self._valid_epoch()
        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)

        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')

        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _train_epoch(self, epoch):
    gc.collect()

    # Fix the feature model and train the inlier model
    self.feat_model.eval()
    self.inlier_model.train()

    # Epoch starts from 1
    total_loss, total_num = 0, 0.0
    data_loader = self.data_loader
    iter_size = self.iter_size

    # Meters for statistics
    average_valid_meter = AverageMeter()
    loss_meter = AverageMeter()
    data_meter = AverageMeter()
    regist_succ_meter = AverageMeter()
    regist_rte_meter = AverageMeter()
    regist_rre_meter = AverageMeter()

    # Timers for profiling
    data_timer = Timer()
    nn_timer = Timer()
    inlier_timer = Timer()
    total_timer = Timer()

    if self.config.num_train_iter > 0:
      num_train_iter = self.config.num_train_iter
    else:
      num_train_iter = len(data_loader) // iter_size
    start_iter = (epoch - 1) * num_train_iter

    tp, fp, tn, fn = 0, 0, 0, 0

    # Iterate over batches
    for curr_iter in range(num_train_iter):
      self.optimizer.zero_grad()

      batch_loss, data_time = 0, 0
      total_timer.tic()

      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = self.get_data(self.train_data_loader_iter)
        data_time += data_timer.toc(average=False)

        # Initial inlier prediction with FCGF and KNN matching
        reg_coords, reg_feats, pred_pairs, is_correct, feat_time, nn_time = self.generate_inlier_input(
            xyz0=input_dict['pcd0'],
            xyz1=input_dict['pcd1'],
            iC0=input_dict['sinput0_C'],
            iC1=input_dict['sinput1_C'],
            iF0=input_dict['sinput0_F'],
            iF1=input_dict['sinput1_F'],
            len_batch=input_dict['len_batch'],
            pos_pairs=input_dict['correspondences'])
        nn_timer.update(nn_time)

        # Inlier prediction with 6D ConvNet
        inlier_timer.tic()
        reg_sinput = ME.SparseTensor(reg_feats.contiguous(),
                                     coords=reg_coords.int()).to(self.device)
        reg_soutput = self.inlier_model(reg_sinput)
        inlier_timer.toc()

        logits = reg_soutput.F
        weights = logits.sigmoid()

        # Truncate weights too low
        # For training, inplace modification is prohibited for backward
        if self.clip_weight_thresh > 0:
          weights_tmp = torch.zeros_like(weights)
          valid_mask = weights > self.clip_weight_thresh
          weights_tmp[valid_mask] = weights[valid_mask]
          weights = weights_tmp

        # Weighted Procrustes
        pred_rots, pred_trans, ws = self.weighted_procrustes(xyz0s=input_dict['pcd0'],
                                                             xyz1s=input_dict['pcd1'],
                                                             pred_pairs=pred_pairs,
                                                             weights=weights)

        # Get batch registration loss
        gt_rots, gt_trans = self.decompose_rotation_translation(input_dict['T_gt'])
        rot_error = batch_rotation_error(pred_rots, gt_rots)
        trans_error = batch_translation_error(pred_trans, gt_trans)
        individual_loss = rot_error + self.config.trans_weight * trans_error

        # Select batches with at least 10 valid correspondences
        valid_mask = ws > 10
        num_valid = valid_mask.sum().item()
        average_valid_meter.update(num_valid)

        # Registration loss against registration GT
        loss = self.config.procrustes_loss_weight * individual_loss[valid_mask].mean()
        if not np.isfinite(loss.item()):
          max_val = loss.item()
          logging.info('Loss is infinite, abort ')
          continue

        # Direct inlier loss against nearest neighbor searched GT
        target = torch.from_numpy(is_correct).squeeze()
        if self.config.inlier_use_direct_loss:
          inlier_loss = self.config.inlier_direct_loss_weight * self.crit(
              logits.cpu().squeeze(), target.to(torch.float)) / iter_size
          loss += inlier_loss

        loss.backward()

        # Update statistics before backprop
        with torch.no_grad():
          regist_rre_meter.update(rot_error.squeeze() * 180 / np.pi)
          regist_rte_meter.update(trans_error.squeeze())

          success = (trans_error.squeeze() < self.config.success_rte_thresh) * (
              rot_error.squeeze() * 180 / np.pi < self.config.success_rre_thresh)
          regist_succ_meter.update(success.float())

          batch_loss += loss.mean().item()

          neg_target = (~target).to(torch.bool)
          pred = logits > 0  # todo thresh
          pred_on_pos, pred_on_neg = pred[target], pred[neg_target]
          tp += pred_on_pos.sum().item()
          fp += pred_on_neg.sum().item()
          tn += (~pred_on_neg).sum().item()
          fn += (~pred_on_pos).sum().item()

          # Check gradient and avoid backprop of inf values
          max_grad = torch.abs(self.inlier_model.final.kernel.grad).max().cpu().item()

        # Backprop only if gradient is finite
        if not np.isfinite(max_grad):
          self.optimizer.zero_grad()
          logging.info(f'Clearing the NaN gradient at iter {curr_iter}')
        else:
          self.optimizer.step()

      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)
      loss_meter.update(batch_loss)

      # Output to logs
      if curr_iter % self.config.stat_freq == 0:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        tpr = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        balanced_accuracy = (tpr + tnr) / 2

        correspondence_accuracy = is_correct.sum() / len(is_correct)

        stat = {
            'loss': loss_meter.avg,
            'precision': precision,
            'recall': recall,
            'tpr': tpr,
            'tnr': tnr,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'num_valid': average_valid_meter.avg,
        }

        for k, v in stat.items():
          self.writer.add_scalar(f'train/{k}', v, start_iter + curr_iter)

        logging.info(' '.join([
            f"Train Epoch: {epoch} [{curr_iter}/{num_train_iter}],",
            f"Current Loss: {loss_meter.avg:.3e},",
            f"Correspondence acc: {correspondence_accuracy:.3e}",
            f", Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f},",
            f"TPR: {tpr:.4f}, TNR: {tnr:.4f}, BAcc: {balanced_accuracy:.4f}",
            f"RTE: {regist_rte_meter.avg:.3e}, RRE: {regist_rre_meter.avg:.3e},",
            f"Succ rate: {regist_succ_meter.avg:3e}",
            f"Avg num valid: {average_valid_meter.avg:3e}",
            f"\tData time: {data_meter.avg:.4f}, Train time: {total_timer.avg - data_meter.avg:.4f},",
            f"NN search time: {nn_timer.avg:.3e}, Total time: {total_timer.avg:.4f}"
        ]))

        loss_meter.reset()
        regist_rte_meter.reset()
        regist_rre_meter.reset()
        regist_succ_meter.reset()
        average_valid_meter.reset()
        data_meter.reset()
        total_timer.reset()

        tp, fp, tn, fn = 0, 0, 0, 0

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.feat_model.eval()
    self.inlier_model.eval()
    self.val_data_loader.dataset.reset_seed(0)

    num_data = 0
    loss_meter = AverageMeter()
    hit_ratio_meter = AverageMeter()
    regist_succ_meter = AverageMeter()
    regist_rte_meter = AverageMeter()
    regist_rre_meter = AverageMeter()
    data_timer = Timer()
    feat_timer = Timer()
    inlier_timer = Timer()
    nn_timer = Timer()
    dgr_timer = Timer()

    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    tot_num_data = int(tot_num_data / self.val_data_loader.batch_size)
    data_loader_iter = self.val_data_loader.__iter__()

    tp, fp, tn, fn = 0, 0, 0, 0
    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = self.get_data(data_loader_iter)
      data_timer.toc()

      reg_coords, reg_feats, pred_pairs, is_correct, feat_time, nn_time = self.generate_inlier_input(
          xyz0=input_dict['pcd0'],
          xyz1=input_dict['pcd1'],
          iC0=input_dict['sinput0_C'],
          iC1=input_dict['sinput1_C'],
          iF0=input_dict['sinput0_F'],
          iF1=input_dict['sinput1_F'],
          len_batch=input_dict['len_batch'],
          pos_pairs=input_dict['correspondences'])
      feat_timer.update(feat_time)
      nn_timer.update(nn_time)

      hit_ratio_meter.update(is_correct.sum().item() / len(is_correct))

      inlier_timer.tic()
      reg_sinput = ME.SparseTensor(reg_feats.contiguous(),
                                   coords=reg_coords.int()).to(self.device)
      reg_soutput = self.inlier_model(reg_sinput)
      inlier_timer.toc()

      dgr_timer.tic()
      logits = reg_soutput.F
      weights = logits.sigmoid()

      if self.clip_weight_thresh > 0:
        weights[weights < self.clip_weight_thresh] = 0

      # Weighted Procrustes
      pred_rots, pred_trans, ws = self.weighted_procrustes(xyz0s=input_dict['pcd0'],
                                                           xyz1s=input_dict['pcd1'],
                                                           pred_pairs=pred_pairs,
                                                           weights=weights)
      dgr_timer.toc()

      valid_mask = ws > 10
      gt_rots, gt_trans = self.decompose_rotation_translation(input_dict['T_gt'])
      rot_error = batch_rotation_error(pred_rots, gt_rots) * 180 / np.pi
      trans_error = batch_translation_error(pred_trans, gt_trans)

      regist_rre_meter.update(rot_error.squeeze())
      regist_rte_meter.update(trans_error.squeeze())

      # Compute success
      success = (trans_error < self.config.success_rte_thresh) * (
          rot_error < self.config.success_rre_thresh) * valid_mask
      regist_succ_meter.update(success.float())

      target = torch.from_numpy(is_correct).squeeze()
      neg_target = (~target).to(torch.bool)
      pred = weights > 0.5  # TODO thresh
      pred_on_pos, pred_on_neg = pred[target], pred[neg_target]
      tp += pred_on_pos.sum().item()
      fp += pred_on_neg.sum().item()
      tn += (~pred_on_neg).sum().item()
      fn += (~pred_on_pos).sum().item()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % self.config.stat_freq == 0:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        tpr = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        balanced_accuracy = (tpr + tnr) / 2
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3e},",
            f"NN search time: {nn_timer.avg:.3e}",
            f"Feature Extraction Time: {feat_timer.avg:.3e}, Inlier Time: {inlier_timer.avg:.3e},",
            f"Loss: {loss_meter.avg:.4f}, Hit Ratio: {hit_ratio_meter.avg:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ",
            f"TPR: {tpr:.4f}, TNR: {tnr:.4f}, BAcc: {balanced_accuracy:.4f}, ",
            f"DGR RTE: {regist_rte_meter.avg:.3e}, DGR RRE: {regist_rre_meter.avg:.3e}, DGR Time: {dgr_timer.avg:.3e}",
            f"DGR Succ rate: {regist_succ_meter.avg:3e}",
        ]))
        data_timer.reset()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    balanced_accuracy = (tpr + tnr) / 2

    logging.info(' '.join([
        f"Feature Extraction Time: {feat_timer.avg:.3e}, NN search time: {nn_timer.avg:.3e}",
        f"Inlier Time: {inlier_timer.avg:.3e}, Final Loss: {loss_meter.avg}, ",
        f"Loss: {loss_meter.avg}, Hit Ratio: {hit_ratio_meter.avg:.4f}, Precision: {precision}, Recall: {recall}, F1: {f1}, ",
        f"TPR: {tpr}, TNR: {tnr}, BAcc: {balanced_accuracy}, ",
        f"RTE: {regist_rte_meter.avg:.3e}, RRE: {regist_rre_meter.avg:.3e}, DGR Time: {dgr_timer.avg:.3e}",
        f"DGR Succ rate: {regist_succ_meter.avg:3e}",
    ]))

    stat = {
        'loss': loss_meter.avg,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'tnr': tnr,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'regist_rte': regist_rte_meter.avg,
        'regist_rre': regist_rre_meter.avg,
        'succ_rate': regist_succ_meter.avg
    }

    return stat

  def _load_weights(self, config):
    if config.resume is None and config.weights:
      logging.info("=> loading weights for inlier model '{}'".format(config.weights))
      checkpoint = torch.load(config.weights)
      self.feat_model.load_state_dict(checkpoint['state_dict'])
      logging.info("=> Loaded base model weights from '{}'".format(config.weights))
      if 'state_dict_inlier' in checkpoint:
        self.inlier_model.load_state_dict(checkpoint['state_dict_inlier'])
        logging.info("=> Loaded inlier weights from '{}'".format(config.weights))
      else:
        logging.warn("Inlier weight not found in '{}'".format(config.weights))

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)

        self.start_epoch = state['epoch']
        self.feat_model.load_state_dict(state['state_dict'])
        self.feat_model = self.feat_model.to(self.device)
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']

        if 'state_dict_inlier' in state:
          self.inlier_model.load_state_dict(state['state_dict_inlier'])
          self.inlier_model = self.inlier_model.to(self.device)
        else:
          logging.warn("Inlier weights not found in '{}'".format(config.resume))
      else:
        logging.warn("Inlier weights does not exist at '{}'".format(config.resume))

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    print('_save_checkpoint from inlier_trainer')
    state = {
        'epoch': epoch,
        'state_dict': self.feat_model.state_dict(),
        'state_dict_inlier': self.inlier_model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

  def get_data(self, iterator):
    while True:
      try:
        input_data = iterator.next()
      except ValueError as e:
        logging.info('Skipping an empty batch')
        continue

      return input_data

  def decompose_by_length(self, tensor, reference_tensors):
    decomposed_tensors = []
    start_ind = 0
    for r in reference_tensors:
      N = len(r)
      decomposed_tensors.append(tensor[start_ind:start_ind + N])
      start_ind += N
    return decomposed_tensors

  def decompose_rotation_translation(self, Ts):
    Ts = Ts.float()
    Rs = Ts[:, :3, :3]
    ts = Ts[:, :3, 3]

    Rs.require_grad = False
    ts.require_grad = False

    return Rs, ts

  def weighted_procrustes(self, xyz0s, xyz1s, pred_pairs, weights):
    decomposed_weights = self.decompose_by_length(weights, pred_pairs)
    RT = []
    ws = []

    for xyz0, xyz1, pred_pair, w in zip(xyz0s, xyz1s, pred_pairs, decomposed_weights):
      xyz0.requires_grad = False
      xyz1.requires_grad = False
      ws.append(w.sum().item())
      predT = GlobalRegistration.weighted_procrustes(
          X=xyz0[pred_pair[:, 0]].to(self.device),
          Y=xyz1[pred_pair[:, 1]].to(self.device),
          w=w,
          eps=np.finfo(np.float32).eps)
      RT.append(predT)

    Rs, ts = list(zip(*RT))
    Rs = torch.stack(Rs, 0)
    ts = torch.stack(ts, 0)
    ws = torch.Tensor(ws)
    return Rs, ts, ws

  def generate_inlier_features(self, xyz0, xyz1, C0, C1, F0, F1, pair_ind0, pair_ind1):
    """
    Assume that the indices 0 and indices 1 gives the pairs in the
    (downsampled) correspondences.
    """
    assert len(pair_ind0) == len(pair_ind1)
    reg_feat_type = self.config.inlier_feature_type
    assert reg_feat_type in ['ones', 'coords', 'counts', 'feats']

    # Move coordinates and indices to the device
    if 'coords' in reg_feat_type:
      C0 = C0.to(self.device)
      C1 = C1.to(self.device)

    # TODO: change it to append the features and then concat at last
    if reg_feat_type == 'ones':
      reg_feat = torch.ones((len(pair_ind0), 1)).to(torch.float32)
    elif reg_feat_type == 'feats':
      reg_feat = torch.cat((F0[pair_ind0], F1[pair_ind1]), dim=1)
    elif reg_feat_type == 'coords':
      reg_feat = torch.cat((torch.cos(torch.cat(
          xyz0, 0)[pair_ind0]), torch.cos(torch.cat(xyz1, 0)[pair_ind1])),
                           dim=1)
    else:
      raise ValueError('Inlier feature type not defined')

    return reg_feat

  def generate_inlier_input(self, xyz0, xyz1, iC0, iC1, iF0, iF1, len_batch, pos_pairs):
    # pairs consist of (xyz1 index, xyz0 index)
    stime = time.time()
    sinput0 = ME.SparseTensor(iF0, coords=iC0).to(self.device)
    oF0 = self.feat_model(sinput0).F

    sinput1 = ME.SparseTensor(iF1, coords=iC1).to(self.device)
    oF1 = self.feat_model(sinput1).F
    feat_time = time.time() - stime

    stime = time.time()
    pred_pairs = self.find_pairs(oF0, oF1, len_batch)
    nn_time = time.time() - stime

    is_correct = find_correct_correspondence(pos_pairs, pred_pairs, len_batch=len_batch)

    cat_pred_pairs = []
    start_inds = torch.zeros((1, 2)).long()
    for lens, pred_pair in zip(len_batch, pred_pairs):
      cat_pred_pairs.append(pred_pair + start_inds)
      start_inds += torch.LongTensor(lens)

    cat_pred_pairs = torch.cat(cat_pred_pairs, 0)
    pred_pair_inds0, pred_pair_inds1 = cat_pred_pairs.t()
    reg_coords = torch.cat((iC0[pred_pair_inds0], iC1[pred_pair_inds1, 1:]), 1)
    reg_feats = self.generate_inlier_features(xyz0, xyz1, iC0, iC1, oF0, oF1,
                                              pred_pair_inds0, pred_pair_inds1).float()

    return reg_coords, reg_feats, pred_pairs, is_correct, feat_time, nn_time

  def find_pairs(self, F0, F1, len_batch):
    nn_batch = find_knn_batch(F0,
                              F1,
                              len_batch,
                              nn_max_n=self.config.nn_max_n,
                              knn=self.config.inlier_knn,
                              return_distance=False,
                              search_method=self.config.knn_search_method)

    pred_pairs = []
    for nns, lens in zip(nn_batch, len_batch):
      pred_pair_ind0, pred_pair_ind1 = torch.arange(
          len(nns)).long()[:, None], nns.long().cpu()
      nn_pairs = []
      for j in range(nns.shape[1]):
        nn_pairs.append(
            torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))

      pred_pairs.append(torch.cat(nn_pairs, 0))
    return pred_pairs
