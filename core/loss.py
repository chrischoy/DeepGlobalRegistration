# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import torch.nn as nn

import numpy as np


class UnbalancedLoss(nn.Module):
  NUM_LABELS = 2

  def __init__(self):
    super().__init__()
    self.crit = nn.BCEWithLogitsLoss()

  def forward(self, logits, label):
    return self.crit(logits, label.to(torch.float))


class BalancedLoss(nn.Module):
  NUM_LABELS = 2

  def __init__(self):
    super().__init__()
    self.crit = nn.BCEWithLogitsLoss()

  def forward(self, logits, label):
    assert torch.all(label < self.NUM_LABELS)
    loss = torch.scalar_tensor(0.).to(logits)
    for i in range(self.NUM_LABELS):
      target_mask = label == i
      if torch.any(target_mask):
        loss += self.crit(logits[target_mask], label[target_mask].to(
            torch.float)) / self.NUM_LABELS
    return loss


class HighDimSmoothL1Loss:

  def __init__(self, weights, quantization_size=1, eps=np.finfo(np.float32).eps):
    self.eps = eps
    self.quantization_size = quantization_size
    self.weights = weights
    if self.weights is not None:
      self.w1 = weights.sum()

  def __call__(self, X, Y):
    sq_dist = torch.sum(((X - Y) / self.quantization_size)**2, axis=1, keepdim=True)
    use_sq_half = 0.5 * (sq_dist < 1).float()

    loss = (0.5 - use_sq_half) * (torch.sqrt(sq_dist + self.eps) -
                                  0.5) + use_sq_half * sq_dist

    if self.weights is None:
      return loss.mean()
    else:
      return (loss * self.weights).sum() / self.w1
