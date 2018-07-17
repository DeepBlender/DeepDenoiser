from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from RenderPasses import RenderPasses

class Naming:

  # Naming for Tensorboard

  @staticmethod
  def difference_name(name, masked=False):
    result = Naming._tensorboard_statistics_name(name, ' Difference', masked=masked)
    return result
  
  @staticmethod
  def mean_name(name, masked=False):
    result = Naming._tensorboard_statistics_name(name, ' Mean', masked=masked)
    return result
  
  @staticmethod
  def variation_difference_name(name, masked=False):
    result = Naming._tensorboard_statistics_name(name, ' Variation Difference', masked=masked)
    return result
  
  @staticmethod
  def variation_mean_name(name, masked=False):
    result = Naming._tensorboard_statistics_name(name, ' Variation Mean', masked=masked)
    return result
  
  @staticmethod
  def ms_ssim_name(name, masked=False):
    result = Naming._tensorboard_statistics_name(name, ' MS SSIM', masked=masked)
    return result
  
  @staticmethod
  def tensorboard_name(name):
    result = name.lower().replace(' ', '_')
    return result
  
  @staticmethod
  def _tensorboard_statistics_name(name, statistics_name, masked=False):
    result = name
    if RenderPasses.is_combined_render_pass(result):
      result = 'Combined ' + result
    result = result + statistics_name
    result = Naming._masked_if_needed(result, masked=masked)
    result = Naming.tensorboard_name(result)
    return result
  
  
  # Naming for tfrecords, statistics, prediction
  
  @staticmethod
  def source_feature_name(name, index=None, masked=False):
    result = 'source_image/'
    if index != None:
      result = result + str(index) + '/'
    result = result + name
    result = Naming._masked_if_needed(result, masked=masked)
    return result
  
  @staticmethod
  def target_feature_name(name, masked=False):
    result = 'target_image/' + name
    result = Naming._masked_if_needed(result, masked=masked)
    return result
  
  @staticmethod
  def prediction_feature_name(name):
    result = 'prediction/' + name
    return result
  
  @staticmethod
  def _masked_if_needed(name, masked):
    result = name
    if masked:
      result = name + ' Masked'
    return result
