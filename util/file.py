# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import re
from os import listdir
from os.path import isfile, isdir, join, splitext

import numpy as np


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, mode=0o755)


def sorted_alphanum(file_list_ordered):
  def convert(text):
    return int(text) if text.isdigit() else text

  def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]

  return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f) for f in listdir(path)
        if isfile(join(path, f)) and splitext(f)[1] == extension
    ]
  file_list = sorted_alphanum(file_list)
  return file_list


def get_file_list_specific(path, color_depth, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f) for f in listdir(path)
        if isfile(join(path, f)) and color_depth in f and splitext(f)[1] == extension
    ]
    file_list = sorted_alphanum(file_list)
  return file_list


def get_folder_list(path):
  folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list


def read_trajectory(filename, dim=4):
  class CameraPose:
    def __init__(self, meta, mat):
      self.metadata = meta
      self.pose = mat

    def __str__(self):
      return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
        "pose : " + "\n" + np.array_str(self.pose)

  traj = []
  with open(filename, 'r') as f:
    metastr = f.readline()
    while metastr:
      metadata = list(map(int, metastr.split()))
      mat = np.zeros(shape=(dim, dim))
      for i in range(dim):
        matstr = f.readline()
        mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
      traj.append(CameraPose(metadata, mat))
      metastr = f.readline()
    return traj
