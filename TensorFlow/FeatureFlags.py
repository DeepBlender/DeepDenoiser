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
    self.feature_flag_names = feature_flag_names
    self.feature_flag_mode = feature_flag_mode
    self.data_format = data_format
    self._render_pass_name_to_feature_flags_names = {}
    self._used_render_pass_names = []
    self._used_feature_flag_names = []
    self._render_pass_name_to_feature_flags = {}
  
  def add_render_pass_name_to_feature_flags_names(self, render_pass_name, feature_flag_names):
    self._render_pass_name_to_feature_flags_names[render_pass_name] = feature_flag_names
    for feature_flag_name in feature_flag_names:
      assert feature_flag_name in self.feature_flag_names
  
  def freeze(self):
    for render_pass_name, feature_flag_names in self._render_pass_name_to_feature_flags_names.items():
      self._used_render_pass_names.append(render_pass_name)
      for feature_flag_name in feature_flag_names:
        if not feature_flag_name in self._used_feature_flag_names:
          self._used_feature_flag_names.append(feature_flag_name)
    self._used_render_pass_names.sort()
    self._used_feature_flag_names.sort()

    shape = [1, 1, 1]
    if self.feature_flag_mode == FeatureFlagMode.FLAGS:
      for render_pass_name in self._used_render_pass_names:
        flags = []
        feature_flag_names = self._render_pass_name_to_feature_flags_names[render_pass_name]
        for used_feature_flag_name in self._used_feature_flag_names:
          if used_feature_flag_name in feature_flag_names:
            flag = tf.ones(shape, tf.float32)
          else:
            flag = tf.zeros(shape, tf.float32)
          flags.append(flag)
        
        if self.data_format == 'channels_last':
          flags = tf.concat(flags, 2)
        else:
          flags = tf.concat(flags, 0)
        self._render_pass_name_to_feature_flags[render_pass_name] = flags

    elif self.feature_flag_mode == FeatureFlagMode.ONE_HOT_ENCODING:
      for render_pass_name in self._used_render_pass_names:
        flags = []
        for used_render_pass_name in self._used_render_pass_names:
          if render_pass_name == used_render_pass_name:
            flag = tf.ones(shape, tf.float32)
          else:
            flag = tf.zeros(shape, tf.float32)
          flags.append(flag)
        
        if self.data_format == 'channels_last':
          flags = tf.concat(flags, 2)
        else:
          flags = tf.concat(flags, 0)
        self._render_pass_name_to_feature_flags[render_pass_name] = flags
    
    elif self.feature_flag_mode == FeatureFlagMode.EMBEDDING:
      self.vocabulary_size = len(self._used_render_pass_names)
      self.embedding_dimension = len(self._used_render_pass_names) // 2

  def feature_flags(self, render_pass_name, height, width, data_format):
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
      index = self._used_render_pass_names.index(render_pass_name)

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

    for render_pass_name in self._used_render_pass_names:
      if (
          self.feature_flag_mode == FeatureFlagMode.FLAGS or
          self.feature_flag_mode == FeatureFlagMode.ONE_HOT_ENCODING):
        
        result = self._render_pass_name_to_feature_flags[render_pass_name]

        # TODO: Tile should happen later on, e.g. in the source encoder
        if self.data_format == 'channels_last':
          result = tf.tile(result, [height, width, 1])
        else:
          result = tf.tile(result, [1, height, width])
        sources[Naming.feature_flags_name(render_pass_name)] = result

