from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from enum import Enum
from Naming import Naming


class FeatureFlagMode(Enum):
  NONE = 1
  FLAGS = 2
  ONE_HOT_ENCODING = 3
  EMBEDDING = 4


class FeatureFlags:

  def __init__(self, feature_flag_names, feature_flag_mode, data_format):
    self.feature_flag_names = sorted(feature_flag_names)
    self.feature_flag_mode = feature_flag_mode
    self.data_format = data_format
    self.feature_flag_name_to_feature_flags = {}

    shape = [1, 1, 1]
    if self.feature_flag_mode == FeatureFlagMode.ONE_HOT_ENCODING:
      for current_feature_flag_name in self.feature_flag_names:
        flags = []
        for feature_flag_name in self.feature_flag_names:
          if current_feature_flag_name == feature_flag_name:
            flag = tf.ones(shape, tf.float32)
          else:
            flag = tf.zeros(shape, tf.float32)
          flags.append(flag)
        
        if self.data_format == 'channels_last':
          flags = tf.concat(flags, 2)
        else:
          flags = tf.concat(flags, 0)
        self.feature_flag_name_to_feature_flags[current_feature_flag_name] = flags
    
    elif self.feature_flag_mode == FeatureFlagMode.EMBEDDING:
      self.vocabulary_size = len(self.feature_flag_names)

      # TODO: Number of dimensions should not be hardcoded.
      self.embedding_dimension = len(self.feature_flag_names) // 2

  def feature_flags(self, feature_flag_name, height, width, data_format):
    assert self.feature_flag_mode == FeatureFlagMode.EMBEDDING

    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
      index = self.feature_flag_names.index(feature_flag_name)

      embedding_matrix = tf.get_variable("feature_flags_embedding_matrix", [self.vocabulary_size, self.embedding_dimension], trainable=True)
      result = tf.nn.embedding_lookup(embedding_matrix, [index])

      if self.data_format == 'channels_last':
        result = tf.reshape(result, [1, 1, self.embedding_dimension])
      else:
        result = tf.reshape(result, [self.embedding_dimension, 1, 1])

      if self.data_format == 'channels_last':
        result = tf.tile(result, [height, width, 1])
      else:
        result = tf.tile(result, [1, height, width])

      return result

  def add_to_source_dictionary(self, sources, height, width):
    if self.feature_flag_mode == FeatureFlagMode.ONE_HOT_ENCODING:
      for feature_flag_name in self.feature_flag_names:
        result = self.feature_flag_name_to_feature_flags[feature_flag_name]

        # TODO: Tile should happen later on, e.g. in the source encoder
        if self.data_format == 'channels_last':
          result = tf.tile(result, [height, width, 1])
        else:
          result = tf.tile(result, [1, height, width])
        sources[Naming.feature_flags_name(feature_flag_name)] = result
