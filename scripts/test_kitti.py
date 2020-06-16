# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import sys
import logging
import argparse
import numpy as np
import open3d as o3d

import torch

from config import get_config

from core.deep_global_registration import DeepGlobalRegistration

from dataloader.kitti_loader import KITTINMPairDataset
from dataloader.base_loader import CollationFunctionFactory
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature, pointcloud_to_spheres
from util.timer import AverageMeter, Timer

from scripts.test_3dmatch import rte_rre

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

TE_THRESH = 0.6  # m
RE_THRESH = 5  # deg
VISUALIZE = False


def visualize_pair(xyz0, xyz1, T, voxel_size):
    pcd0 = pointcloud_to_spheres(xyz0,
                                 voxel_size,
                                 np.array([0, 0, 1]),
                                 sphere_size=0.6)
    pcd1 = pointcloud_to_spheres(xyz1,
                                 voxel_size,
                                 np.array([0, 1, 0]),
                                 sphere_size=0.6)
    pcd0.transform(T)
    o3d.visualization.draw_geometries([pcd0, pcd1])


def analyze_stats(stats):
    print('Total result mean')
    print(stats.mean(0))

    sel_stats = stats[stats[:, 0] > 0]
    print(sel_stats.mean(0))


def evaluate(config, data_loader, method):
    data_timer = Timer()

    test_iter = data_loader.__iter__()
    N = len(test_iter)

    stats = np.zeros((N, 5))  # bool succ, rte, rre, time, drive id

    for i in range(len(data_loader)):
        data_timer.tic()
        try:
            data_dict = test_iter.next()
        except ValueError as exc:
            pass
        data_timer.toc()

        drive = data_dict['extra_packages'][0]['drive']
        xyz0, xyz1 = data_dict['pcd0'][0], data_dict['pcd1'][0]
        T_gt = data_dict['T_gt'][0].numpy()
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        T_pred = method.register(xyz0np, xyz1np)

        stats[i, :3] = rte_rre(T_pred, T_gt, TE_THRESH, RE_THRESH)
        stats[i, 3] = method.reg_timer.diff + method.feat_timer.diff
        stats[i, 4] = drive

        if stats[i, 0] == 0:
            logging.info(f"Failed with RTE: {stats[i, 1]}, RRE: {stats[i, 2]}")

        if i % 10 == 0:
            succ_rate, rte, rre, avg_time, _ = stats[:i + 1].mean(0)
            logging.info(
                f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg},"
                + f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
                f" RRE: {rre}, Success: {succ_rate * 100} %")

        if VISUALIZE and i % 10 == 9:
            visualize_pair(xyz0, xyz1, T_pred, config.voxel_size)

    succ_rate, rte, rre, avg_time, _ = stats.mean(0)
    logging.info(
        f"Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg}," +
        f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
        f" RRE: {rre}, Success: {succ_rate * 100} %")

    # Save results
    filename = f'kitti-stats_{method.__class__.__name__}'
    if config.out_filename is not None:
        filename += f'_{config.out_filename}'
    if isinstance(method, FCGFWrapper):
        filename += '_' + method.method
        if 'ransac' in method.method:
            filename += f'_{config.ransac_iter}'
    if os.path.isdir(config.out_dir):
        out_file = os.path.join(config.out_dir, filename)
    else:
        out_file = filename  # save it on the current directory
    print(f'Saving the stats to {out_file}')
    np.savez(out_file, stats=stats)
    analyze_stats(stats)


if __name__ == '__main__':
    config = get_config()

    dgr = DeepGlobalRegistration(config)

    dset = KITTINMPairDataset('test',
                              transform=None,
                              random_rotation=False,
                              random_scale=False,
                              config=config)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair'),
        pin_memory=False,
        drop_last=False)

    evaluate(config, data_loader, dgr)
