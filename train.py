# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import open3d as o3d  # prevent loading error

import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from config import get_config

from dataloader.data_loaders import make_data_loader

from core.trainer import WeightedProcrustesTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def main(config, resume=False):
  train_loader = make_data_loader(config,
                                  config.train_phase,
                                  config.batch_size,
                                  num_workers=config.train_num_workers,
                                  shuffle=True)

  if config.test_valid:
    val_loader = make_data_loader(config,
                                  config.val_phase,
                                  config.val_batch_size,
                                  num_workers=config.val_num_workers,
                                  shuffle=True)
  else:
    val_loader = None

  trainer = WeightedProcrustesTrainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
  )

  trainer.train()


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config()

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
