# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
from urllib.request import urlretrieve

import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

BASE_URL = "http://node2.chrischoy.org/data/"
DOWNLOAD_LIST = [
    (BASE_URL + "datasets/registration/", "redkitchen_000.ply"),
    (BASE_URL + "datasets/registration/", "redkitchen_010.ply"),
    (BASE_URL + "projects/DGR/", "ResUNetBN2C-feat32-3dmatch-v0.05.pth")
]

# Check if the weights and file exist and download
if not os.path.isfile('redkitchen_000.ply'):
  print('Downloading weights and pointcloud files...')
  for f in DOWNLOAD_LIST:
    print(f"Downloading {f}")
    urlretrieve(f[0] + f[1], f[1])

if __name__ == '__main__':
  config = get_config()
  if config.weights is None:
    config.weights = DOWNLOAD_LIST[-1][-1]

  # preprocessing
  pcd0 = o3d.io.read_point_cloud(config.pcd0)
  pcd0.estimate_normals()
  pcd1 = o3d.io.read_point_cloud(config.pcd1)
  pcd1.estimate_normals()

  # registration
  dgr = DeepGlobalRegistration(config)
  T01 = dgr.register(pcd0, pcd1)

  o3d.visualization.draw_geometries([pcd0, pcd1])

  pcd0.transform(T01)
  print(T01)

  o3d.visualization.draw_geometries([pcd0, pcd1])
