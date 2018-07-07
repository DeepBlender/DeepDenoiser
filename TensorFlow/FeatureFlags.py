from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeatureFlags:

  def __init__(self, feature_flag_names):
    self.feature_flag_names = feature_flag_names
    self.render_pass_name_to_feature_flag_names = {}
    self.used_feature_flag_names = []
  
  def add_render_pass_name_to_feature_flag_names(self, render_pass_name, feature_flag_names):
    self.render_pass_name_to_feature_flag_names[render_pass_name] = feature_flag_names
  
  def freeze(self):
    for feature_flag_name in self.feature_flag_names:
      for _, used_feature_flag_names in self.render_pass_name_to_feature_flag_names.items():
        if feature_flag_name in used_feature_flag_names:
          self.used_feature_flag_names.append(feature_flag_name)
          break
  
  def feature_flags(self, render_pass_name, height, width, data_format='channels_last'):
    feature_flag_names = self.render_pass_name_to_feature_flag_names[render_pass_name]
    result = []
    for used_feature_flag_name in self.used_feature_flag_names:
      if data_format == 'channels_last':
        shape = [height, width, 1]
      else:
        shape = [1, height, width]
      
      if used_feature_flag_name in feature_flag_names:
        flag = tf.ones(shape, tf.float32)
      else:
        flag = tf.zeros(shape, tf.float32)
      result.append(flag)
    
    if data_format == 'channels_last':
      result = tf.concat(result, 3)
    else:
      result = tf.concat(result, 0)
    
    return result
