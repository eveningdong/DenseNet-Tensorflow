from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim

# DenseNet 264 in the paper should be a typo, since the number of  layers can not be odd, take 121, 169 and 201 as reference.
networks = {
    'densenet_121': [6, 12, 24, 16],
    'densenet_169': [6, 12, 32, 32],
    'densenet_201': [6, 12, 48, 32],
    'densenet_265': [6, 12, 64, 48], 
}

def dense_arg_scope(weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    use_batch_norm=True):
  """Defines the default DenseNet arg scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    use_batch_norm: Whether or not to use batch normalization.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope([slim.batch_norm], **batch_norm_params):
    with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.contrib.layers.xavier_initializer(),
      activation_fn=None,
      normalizer_fn=None):
      with slim.arg_scope([slim.avg_pool2d], padding='SAME') as arg_sc:
        return arg_sc