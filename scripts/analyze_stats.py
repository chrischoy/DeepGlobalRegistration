# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import argparse

PROPERTY_IDX_MAP = {
    'Recall': 0,
    'TE (m)': 1,
    'RE (deg)': 2,
    'log Time (s)': 3,
    'Scene ID': 4
}


def analyze_by_pair(stats, rte_thresh, rre_thresh):
  '''
  \input stats: (num_methods, num_pairs, num_pairwise_stats=5)
  \return valid mean_stats: (num_methods, 4)
  4 properties: recall, rte, rre, time
  '''
  num_methods, num_pairs, num_pairwise_stats = stats.shape
  pairwise_stats = np.zeros((num_methods, 4))

  for m in range(num_methods):
    # Filter valid registrations by rte / rre thresholds
    mask_rte = stats[m, :, 1] < rte_thresh
    mask_rre = stats[m, :, 2] < rre_thresh
    mask_valid = mask_rte * mask_rre

    # Recall, RTE, RRE, Time
    pairwise_stats[m, 0] = mask_valid.mean()
    pairwise_stats[m, 1] = stats[m, mask_valid, 1].mean()
    pairwise_stats[m, 2] = stats[m, mask_valid, 2].mean()
    pairwise_stats[m, 3] = stats[m, mask_valid, 3].mean()

  return pairwise_stats


def analyze_by_scene(stats, scene_id_list, rte_thresh=0.3, rre_thresh=15):
  '''
  \input stats: (num_methods, num_pairs, num_pairwise_stats=5)
  \return scene_wise mean stats: (num_methods, num_scenes, 4)
  4 properties: recall, rte, rre, time
  '''
  num_methods, num_pairs, num_pairwise_stats = stats.shape
  num_scenes = len(scene_id_list)

  scene_wise_stats = np.zeros((num_methods, len(scene_id_list), 4))

  for m in range(num_methods):
    # Filter valid registrations by rte / rre thresholds
    mask_rte = stats[m, :, 1] < rte_thresh
    mask_rre = stats[m, :, 2] < rre_thresh
    mask_valid = mask_rte * mask_rre

    for s in scene_id_list:
      mask_scene = stats[m, :, 4] == s

      # Valid registrations in the scene
      mask = mask_scene * mask_valid

      # Recall, RTE, RRE, Time
      scene_wise_stats[m, s, 0] = 0 if np.sum(mask_scene) == 0 else float(
          np.sum(mask)) / float(np.sum(mask_scene))
      scene_wise_stats[m, s, 1] = stats[m, mask, 1].mean()
      scene_wise_stats[m, s, 2] = stats[m, mask, 2].mean()
      scene_wise_stats[m, s, 3] = stats[m, mask, 3].mean()

  return scene_wise_stats


def plot_precision_recall_curves(stats, method_names, rte_precisions, rre_precisions,
                                 output_postfix, cmap):
  '''
  \input stats: (num_methods, num_pairs, 5)
  \input method_names:  (num_methods) string, shown as xticks
  '''
  num_methods, num_pairs, _ = stats.shape
  rre_precision_curves = np.zeros((num_methods, len(rre_precisions)))
  rte_precision_curves = np.zeros((num_methods, len(rte_precisions)))

  for i, rre_thresh in enumerate(rre_precisions):
    pairwise_stats = analyze_by_pair(stats, rte_thresh=np.inf, rre_thresh=rre_thresh)
    rre_precision_curves[:, i] = pairwise_stats[:, 0]

  for i, rte_thresh in enumerate(rte_precisions):
    pairwise_stats = analyze_by_pair(stats, rte_thresh=rte_thresh, rre_thresh=np.inf)
    rte_precision_curves[:, i] = pairwise_stats[:, 0]

  fig = plt.figure(figsize=(10, 3))
  ax1 = fig.add_subplot(1, 2, 1, aspect=3.0 / np.max(rte_precisions))
  ax2 = fig.add_subplot(1, 2, 2, aspect=3.0 / np.max(rre_precisions))

  for m, name in enumerate(method_names):
    alpha = rre_precision_curves[m].mean()
    alpha = 1.0 if alpha > 0 else 0.0
    ax1.plot(rre_precisions, rre_precision_curves[m], color=cmap[m], alpha=alpha)
    ax2.plot(rte_precisions, rte_precision_curves[m], color=cmap[m], alpha=alpha)

  ax1.set_ylabel('Recall')
  ax1.set_xlabel('Rotation (deg)')
  ax1.set_ylim((0.0, 1.0))

  ax2.set_xlabel('Translation (m)')
  ax2.set_ylim((0.0, 1.0))
  ax2.legend(method_names, loc='center left', bbox_to_anchor=(1, 0.5))
  ax1.grid()
  ax2.grid()

  plt.tight_layout()
  plt.savefig('{}_{}.png'.format('precision_recall', output_postfix))

  plt.close(fig)


def plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, property_name,
                          ylim, output_postfix, cmap):
  '''
  \input scene_wise_stats: (num_methods, num_scenes, 4)
  \input method_names:  (num_methods) string, shown as xticks
  \input scene_names:   (num_scenes) string, shown as legends
  \input property_name: string, shown as ylabel
  '''
  num_methods, num_scenes, _ = scene_wise_stats.shape
  assert len(method_names) == num_methods
  assert len(scene_names) == num_scenes

  # Initialize figure
  fig = plt.figure(figsize=(14, 3))
  ax = fig.add_subplot(1, 1, 1)

  # Add some paddings
  w = 1.0 / (num_methods + 2)

  # Rightmost bar
  x = np.arange(0, num_scenes) - 0.5 * w * num_methods

  for m in range(num_methods):
    m_stats = scene_wise_stats[m, :, PROPERTY_IDX_MAP[property_name]]
    valid = not (np.logical_and.reduce(np.isnan(m_stats))
                 or np.logical_and.reduce(m_stats == 0))
    alpha = 1.0 if valid else 0.0
    ax.bar(x + m * w, m_stats, w, color=cmap[m], alpha=alpha)

  plt.ylim(ylim)
  plt.xlim((0 - w * num_methods, num_scenes))
  plt.ylabel(property_name)
  plt.xticks(np.arange(0, num_scenes), tuple(scene_names))
  ax.legend(method_names, loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()
  plt.grid()
  plt.savefig('{}_{}.png'.format(property_name, output_postfix))
  plt.close(fig)


def plot_pareto_frontier(pairwise_stats, method_names, cmap):
  recalls = pairwise_stats[:, 0]
  times = 1.0 / pairwise_stats[:, 3]

  ind = np.argsort(times)

  offset = 0.05
  plt.rcParams.update({'font.size': 30})

  fig = plt.figure(figsize=(20, 12))
  ax = fig.add_subplot(111)
  ax.set_xlabel('Number of registrations per second (log scale)')
  ax.set_xscale('log')
  xmin = np.power(10, -2.2)
  xmax = np.power(10, 1.5)
  ax.set_xlim(xmin, xmax)

  ax.set_ylabel('Registration recall')
  ax.set_ylim(-offset, 1)
  ax.set_yticks(np.arange(0, 1, step=0.2))

  plots = [None for m in ind]
  max_gain = -1
  for m in ind[::-1]:
    # 8, 9: our methods
    if (recalls[m] > max_gain):
        max_gain = recalls[m]
        ax.add_patch(
          Rectangle((0, -offset),
                    times[m],
                    recalls[m] + offset,
                    facecolor=(0.94, 0.94, 0.94)))

    plot, = ax.plot(times[m], recalls[m], 'o', c=colors[m], markersize=30)
    plots[m] = plot

  ax.legend(plots, method_names, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  plt.savefig('frontier.png')


if __name__ == '__main__':
  '''
  Input .npz file to analyze:
  \prop npz['stats']: (num_methods, num_pairs, num_pairwise_stats=5)
  5 pairwise stats properties consist of
  - \bool success: decided by evaluation thresholds, will be ignored in this script
  - \float rte: relative translation error (in cm)
  - \float rre: relative rotation error (in deg)
  - \float time: registration time for the pair (in ms)
  - \int scene_id: specific for 3DMatch test sets (8 scenes in total)

  \prop npz['names']: (num_methods)
  Corresponding method name stored in string
  '''

  # Setup fonts
  from matplotlib import rc
  rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
  rc('text', usetex=False)

  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('npz', help='path to the npz file')
  parser.add_argument('--output_postfix', default='', help='postfix of the output')
  parser.add_argument('--end_method_index',
                      default=1000,
                      type=int,
                      help='reserved only for making slides')
  args = parser.parse_args()

  # Load npz file with aformentioned format
  npz = np.load(args.npz)
  stats = npz['stats']

  # Reserved only for making slides, will be skipped by default
  stats[args.end_method_index:, :, 1] = np.inf
  stats[args.end_method_index:, :, 2] = np.inf

  method_names = npz['names']
  scene_names = [
      'Kitchen', 'Home1', 'Home2', 'Hotel1', 'Hotel2', 'Hotel3', 'Study', 'Lab'
  ]

  cmap = plt.get_cmap('tab20b')
  colors = [cmap(i) for i in np.linspace(0, 1, len(method_names))]
  colors.reverse()

  # Plot scene-wise bar charts
  scene_wise_stats = analyze_by_scene(stats,
                                      range(len(scene_names)),
                                      rte_thresh=0.3,
                                      rre_thresh=15)

  plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'Recall',
                        (0.0, 1.0), args.output_postfix, colors)
  plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'TE (m)',
                        (0.0, 0.3), args.output_postfix, colors)
  plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'RE (deg)',
                        (0.0, 15.0), args.output_postfix, colors)

  # Plot rte/rre - recall curves
  plot_precision_recall_curves(stats,
                               method_names,
                               rre_precisions=np.arange(0, 15, 0.05),
                               rte_precisions=np.arange(0, 0.3, 0.005),
                               output_postfix=args.output_postfix,
                               cmap=colors)

  pairwise_stats = analyze_by_pair(stats, rte_thresh=0.3, rre_thresh=15)
  plot_pareto_frontier(pairwise_stats, method_names, cmap=colors)
