# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import open3d as o3d
import argparse
import os, sys
import numpy as np


def read_rgbd_image(color_file, depth_file, max_depth=4.5):
  '''
  \return RGBD image
  '''
  color = o3d.io.read_image(color_file)
  depth = o3d.io.read_image(depth_file)

  # We assume depth scale is always 1000.0
  rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
      color, depth, depth_trunc=max_depth, convert_rgb_to_intensity=False)
  return rgbd_image


def read_pose(pose_file):
  '''
  \return 4x4 np matrix
  '''
  pose = np.loadtxt(pose_file)
  assert pose is not None
  return pose


def read_intrinsics(intrinsic_file):
  '''
  \return fx, fy, cx, cy
  '''
  K = np.loadtxt(intrinsic_file)
  assert K is not None
  return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def integrate_rgb_frames_for_fragment(color_files,
                                      depth_files,
                                      pose_files,
                                      seq_path,
                                      intrinsic,
                                      fragment_id,
                                      n_fragments,
                                      n_frames_per_fragment,
                                      voxel_length=0.008):
  volume = o3d.integration.ScalableTSDFVolume(
      voxel_length=voxel_length,
      sdf_trunc=0.04,
      color_type=o3d.integration.TSDFVolumeColorType.RGB8)

  start = fragment_id * n_frames_per_fragment
  end = min(start + n_frames_per_fragment, len(pose_files))
  for i_abs in range(start, end):
    print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
          (fragment_id, n_fragments - 1, i_abs, i_abs - start + 1, end - start))

    rgbd = read_rgbd_image(
        os.path.join(seq_path, color_files[i_abs]),
        os.path.join(seq_path, depth_files[i_abs]))
    pose = read_pose(os.path.join(seq_path, pose_files[i_abs]))
    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

  mesh = volume.extract_triangle_mesh()
  return mesh


def process_seq(seq_path, output_path, n_frames_per_fragment, display=False):
  files = os.listdir(seq_path)

  if 'intrinsics.txt' in files:
    fx, fy, cx, cy = read_intrinsics(os.path.join(seq_path, 'intrinsics.txt'))
  else:
    fx, fy, cx, cy = read_intrinsics(os.path.join(seq_path, '../camera-intrinsics.txt'))

  rgb_files = sorted(list(filter(lambda x: x.endswith('.color.png'), files)))
  depth_files = sorted(list(filter(lambda x: x.endswith('.depth.png'), files)))
  pose_files = sorted(list(filter(lambda x: x.endswith('.pose.txt'), files)))

  assert len(rgb_files) > 0
  assert len(rgb_files) == len(depth_files)
  assert len(rgb_files) == len(pose_files)

  # get width and height to prepare for intrinsics
  rgb_sample = o3d.io.read_image(os.path.join(seq_path, rgb_files[0]))
  width, height = rgb_sample.get_max_bound()
  intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)

  n_fragments = ((len(rgb_files) + n_frames_per_fragment - 1)) // n_frames_per_fragment

  for fragment_id in range(0, n_fragments):
    mesh = integrate_rgb_frames_for_fragment(rgb_files, depth_files, pose_files,
                                             seq_path, intrinsic, fragment_id,
                                             n_fragments, n_frames_per_fragment)
    if display:
      o3d.visualization.draw_geometries([mesh])

    mesh_name = os.path.join(output_seq_path, 'fragment-{}.ply'.format(fragment_id))
    o3d.io.write_triangle_mesh(mesh_name, mesh)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='RGB-D integration for 3DMatch raw dataset')

  parser.add_argument(
      'dataset', help='path to dataset that contains colors, depths and poses')
  parser.add_argument('output', help='path to output fragments')

  args = parser.parse_args()

  scene_name = args.dataset.split('/')[-1]
  if not os.path.exists(args.output):
    os.makedirs(args.output)

  output_scene_path = os.path.join(args.output, scene_name)
  if os.path.exists(output_scene_path):
    choice = input(
        'Path {} already exists, continue? (Y / N)'.format(output_scene_path))
    if choice != 'Y' and choice != 'y':
      print('abort')
      exit
  else:
    os.makedirs(output_scene_path)

  seqs = list(filter(lambda x: x.startswith('seq'), os.listdir(args.dataset)))
  for seq in seqs:
    output_seq_path = os.path.join(output_scene_path, seq)
    if not os.path.exists(output_seq_path):
      os.makedirs(output_seq_path)
    process_seq(
        os.path.join(args.dataset, seq),
        output_seq_path,
        n_frames_per_fragment=50,
        display=False)
