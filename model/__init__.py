# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import logging
import model.simpleunet as simpleunets
import model.resunet as resunets
import model.pyramidnet as pyramids

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a or 'MLP' in a])


add_models(simpleunets)
add_models(resunets)
add_models(pyramids)


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = MODELS
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    logging.info(f'Invalid model index. You put {name}. Options are:')
    # Display a list of valid model names
    for model in all_models:
      logging.info('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
