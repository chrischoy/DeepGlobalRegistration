# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# yapf: disable
logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='WeightedProcrustesTrainer')

# Batch setting
trainer_arg.add_argument('--batch_size', type=int, default=4)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)

trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--save_epoch_freq', type=int, default=1)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)

trainer_arg.add_argument('--stat_freq', type=int, default=40, help='Frequency for writing stats to log')
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)


trainer_arg.add_argument('--use_balanced_loss', type=str2bool, default=False)
trainer_arg.add_argument('--inlier_direct_loss_weight', type=float, default=1.)
trainer_arg.add_argument('--procrustes_loss_weight', type=float, default=1.)
trainer_arg.add_argument('--trans_weight', type=float, default=1)

trainer_arg.add_argument('--eval_registration', type=str2bool, default=True)
trainer_arg.add_argument('--clip_weight_thresh', type=float, default=0.05, help='Weight threshold for detecting inliers')
trainer_arg.add_argument('--best_val_metric', type=str, default='succ_rate')

# Inlier detection trainer
inlier_arg = add_argument_group('Inlier')
inlier_arg.add_argument('--inlier_model', type=str, default='ResUNetBN2C')
inlier_arg.add_argument('--inlier_feature_type', type=str, default='ones')
inlier_arg.add_argument('--inlier_conv1_kernel_size', type=int, default=3)
inlier_arg.add_argument('--inlier_knn', type=int, default=1)
inlier_arg.add_argument('--knn_search_method', type=str, default='gpu')
inlier_arg.add_argument('--inlier_use_direct_loss', type=str2bool, default=True)

# Feature specific configurations
feat_arg = add_argument_group('feat')
feat_arg.add_argument('--feat_model', type=str, default='SimpleNetBN2C')
feat_arg.add_argument('--feat_model_n_out', type=int, default=16, help='Feature dimension')
feat_arg.add_argument('--feat_conv1_kernel_size', type=int, default=3)
feat_arg.add_argument('--normalize_feature', type=str2bool, default=True)
feat_arg.add_argument('--use_xyz_feature', type=str2bool, default=False)
feat_arg.add_argument('--dist_type', type=str, default='L2')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--num_train_iter', type=int, default=-1, help='train N iter if positive')
opt_arg.add_argument('--icp_cache_path', type=str, default="icp")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_workers', type=int, default=2)
misc_arg.add_argument('--val_num_workers', type=int, default=1)
misc_arg.add_argument('--test_num_workers', type=int, default=2)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument('--nn_max_n', type=int, default=250, help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset03')
data_arg.add_argument('--voxel_size', type=float, default=0.025)
data_arg.add_argument('--threed_match_dir', type=str, default='.')
data_arg.add_argument('--kitti_dir', type=str, default=None, help="Path to the KITTI odometry dataset. This path should contain <kitti_dir>/dataset/sequences.")
data_arg.add_argument('--kitti_max_time_diff', type=int, default=3, help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')

# Evaluation
eval_arg = add_argument_group('Data')
eval_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)
eval_arg.add_argument('--success_rte_thresh', type=float, default=0.3, help='Success if the RTE below this (m)')
eval_arg.add_argument('--success_rre_thresh', type=float, default=15, help='Success if the RTE below this (degree)')
eval_arg.add_argument('--test_random_crop', action='store_true')
eval_arg.add_argument('--test_random_rotation', type=str2bool, default=False)

# Demo
demo_arg = add_argument_group('demo')
demo_arg.add_argument('--pcd0', default="redkitchen_000.ply", type=str)
demo_arg.add_argument('--pcd1', default="redkitchen_010.ply", type=str)
# yapf: enable


def get_config():
  args = parser.parse_args()
  return args
