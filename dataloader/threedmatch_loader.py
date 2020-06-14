# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import glob

from dataloader.base_loader import *
from dataloader.transforms import *

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
from util.file import read_trajectory


class IndoorPairDataset(PairDataset):
  '''
  Train dataset
  '''
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.threed_match_dir
    self.use_xyz_feature = config.use_xyz_feature
    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
      fnames_txt = glob.glob(root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
          self.files.append([fname[0], fname[1]])

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Select features and points using the returned voxelized indices
    # 3DMatch color is not helpful
    # pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    # pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    # xyz as feats
    if self.use_xyz_feature:
      feats_train0.append(unique_xyz0_th - unique_xyz0_th.mean(0))
      feats_train1.append(unique_xyz1_th - unique_xyz1_th.mean(0))
    else:
      feats_train0.append(torch.ones((npts0, 1)))
      feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    extra_package = {'idx': idx, 'file0': file0, 'file1': file1}

    return (unique_xyz0_th.float(),
            unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
            feats1.float(), matches, trans, extra_package)


class ThreeDMatchPairDataset03(IndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './dataloader/split/train_3dmatch.txt',
      'val': './dataloader/split/val_3dmatch.txt',
      'test': './dataloader/split/test_3dmatch.txt'
  }


class ThreeDMatchPairDataset05(ThreeDMatchPairDataset03):
  OVERLAP_RATIO = 0.5


class ThreeDMatchPairDataset07(ThreeDMatchPairDataset03):
  OVERLAP_RATIO = 0.7


class ThreeDMatchTrajectoryDataset(PairDataset):
  '''
  Test dataset
  '''
  DATA_FILES = {
      'train': './dataloader/split/train_3dmatch.txt',
      'val': './dataloader/split/val_3dmatch.txt',
      'test': './dataloader/split/test_3dmatch.txt'
  }

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               scene_id=None,
               config=None,
               return_ply_names=False):

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    self.root = config.threed_match_dir

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if scene_id is not None:
      subset_names = [subset_names[scene_id]]
    for sname in subset_names:
      traj_file = os.path.join(self.root, sname + '-evaluation/gt.log')
      assert os.path.exists(traj_file)
      traj = read_trajectory(traj_file)
      for ctraj in traj:
        i = ctraj.metadata[0]
        j = ctraj.metadata[1]
        T_gt = ctraj.pose
        self.files.append((sname, i, j, T_gt))

    self.return_ply_names = return_ply_names

  def __getitem__(self, pair_index):
    sname, i, j, T_gt = self.files[pair_index]
    ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{i}.ply')
    ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{j}.ply')

    if self.return_ply_names:
      return sname, ply_name0, ply_name1, T_gt

    pcd0 = o3d.io.read_point_cloud(ply_name0)
    pcd1 = o3d.io.read_point_cloud(ply_name1)
    pcd0 = np.asarray(pcd0.points)
    pcd1 = np.asarray(pcd1.points)
    return sname, pcd0, pcd1, T_gt
