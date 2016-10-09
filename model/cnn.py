"""
Convolutional Neural Network
---
@author TaoPR (github.com/starcolon)
"""

import numpy as np
from theano import *
from theano import tensor as T
from theano.tensor import tensor4, lvector, mean, neq
from opendeep import config_root_logger
from opendeep.data import ModifyStream
from opendeep.models import Prototype, Conv2D, Dense, Softmax
from opendeep.models.utils import Pool2D, Flatten
from opendeep.monitor import Monitor, FileService
from opendeep.optimization.loss import MSE
from opendeep.optimization import SGD, AdaDelta

from termcolor import colored
from . import *

class CNN():

  """
  @param {(int,int)} dimension of feature vector (w,h)
  @param {int} dimension of the final vector
  @param {str} working directory
  """
  def __init__(self, image_dim, final_vec_dim, dirw):
    img_shape = (3,) + image_dim
    img_var   = tensor4('xs')

    # Register working directory
    self.dir  = dirw

    self.net  = Prototype()
    # First convnet layer
    self.net.add(Conv2D(inputs=(img_shape, img_var)),n_filters=100, filter_size=(5,5))
    self.net.add(Pool2D,size=(3,3))
    # Second convnet layer
    self.net.add(Conv2D(n_filters=50, filter_size=(3,3)))
    self.net.add(Pool2D,size=(2,2))
    # 3D => 2D flatten
    self.net.add(Flatten,ndim=2)
    # Ordinary hidden layers
    self.net.add(Dense(outputs=128, activation='relu'))
    self.net.add(Dense(outputs=40, activation='relu'))


  def train(self,X,y,epoch):
    target_v  = dvector('y')
    loss      = MSE(
      inputs=self.net.models[-1].p_y_given_x,
      targets=y,
      one_hot=False)

    error_monitor = Monitor(
      name='error', 
      expression=mean(neq(lenet.models[-1].y_pred, y)), 
      valid=True, test=True, 
      out_service=FileService(self.dirw + '/train_error.txt'))

    optimiser = AdaDelta(model=self.net, loss=loss, dataset=X, epoch=epoch)
    optimiser.train(monitor_channels=error_monitor)

  def predict(self,candidates):
    return self.net.predict(candidates)



