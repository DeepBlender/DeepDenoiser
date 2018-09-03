from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import multiprocessing

import math
import json

from Naming import Naming
from RenderPasses import RenderPasses
from FeatureStatistics import Statistics
from FeatureStatistics import FeatureStatistics
import Utilities
from Conv2dUtilities import Conv2dUtilities


class TFRecordsStatistics:
  def __init__(self, tfrecords_creator):
    self.tfrecords_creator = tfrecords_creator
  
  def compute_and_save_statistics(self):

    source_samples_per_pixel_lists = []
    if self.tfrecords_creator.group_by_samples_per_pixel:
      for source_samples_per_pixel_list in self.tfrecords_creator.source_samples_per_pixel_list:
        source_samples_per_pixel_lists.append([source_samples_per_pixel_list])
    else:
      source_samples_per_pixel_lists.append(self.tfrecords_creator.source_samples_per_pixel_list)

    for source_samples_per_pixel_list in source_samples_per_pixel_lists:
      statistics = {}
      statistics['tiles_height_width'] = self.tfrecords_creator.tiles_height_width
      statistics['number_of_sources_per_example'] = self.tfrecords_creator.number_of_sources_per_example
      statistics['source_samples_per_pixel_list'] = source_samples_per_pixel_list

      samples_per_pixel = None
      if self.tfrecords_creator.group_by_samples_per_pixel:
        assert len(source_samples_per_pixel_list) == 1
        samples_per_pixel = source_samples_per_pixel_list[0]
    
      # Prepare the data structures.
    
      self.minimums = {}
      self.maximums = {}
      self.means = {}
      self.variances = {}
      self.minimums_log1p = {}
      self.maximums_log1p = {}
      self.means_log1p = {}
      self.variances_log1p = {}
      
      for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
        source_feature_name = Naming.source_feature_name(source_render_pass)
        self.minimums[source_feature_name] = math.inf
        self.maximums[source_feature_name] = -math.inf
        self.means[source_feature_name] = []
        self.variances[source_feature_name] = []
        self.minimums_log1p[source_feature_name] = math.inf
        self.maximums_log1p[source_feature_name] = -math.inf
        self.means_log1p[source_feature_name] = []
        self.variances_log1p[source_feature_name] = []
        
        if (
            RenderPasses.is_direct_or_indirect_render_pass(source_render_pass) and not
            RenderPasses.is_volume_render_pass(source_render_pass)):
          source_feature_name = Naming.source_feature_name(source_render_pass, masked=True)
          self.minimums[source_feature_name] = math.inf
          self.maximums[source_feature_name] = -math.inf
          self.means[source_feature_name] = []
          self.variances[source_feature_name] = []
          self.minimums_log1p[source_feature_name] = math.inf
          self.maximums_log1p[source_feature_name] = -math.inf
          self.means_log1p[source_feature_name] = []
          self.variances_log1p[source_feature_name] = []
      
      for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
        target_feature_name = Naming.target_feature_name(target_render_pass)
        self.minimums[target_feature_name] = math.inf
        self.maximums[target_feature_name] = -math.inf
        self.means[target_feature_name] = []
        self.variances[target_feature_name] = []
        self.minimums_log1p[target_feature_name] = math.inf
        self.maximums_log1p[target_feature_name] = -math.inf
        self.means_log1p[target_feature_name] = []
        self.variances_log1p[target_feature_name] = []
        
        if (
            RenderPasses.is_direct_or_indirect_render_pass(target_render_pass) and not
            RenderPasses.is_volume_render_pass(target_render_pass)):
          target_feature_name = Naming.target_feature_name(target_render_pass, masked=True)
          self.minimums[target_feature_name] = math.inf
          self.maximums[target_feature_name] = -math.inf
          self.means[target_feature_name] = []
          self.variances[target_feature_name] = []
          self.minimums_log1p[target_feature_name] = math.inf
          self.maximums_log1p[target_feature_name] = -math.inf
          self.means_log1p[target_feature_name] = []
          self.variances_log1p[target_feature_name] = []
      
      
      # Iterate through the tfrecords to compute the usual and log1p statistics for minimum, maximum and mean.
      
      iterator = self._dataset_iterator(samples_per_pixel)
      while True:
        try:
          source_features, target_features = iterator.get_next()
          
          for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
            for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
              source_feature_name = Naming.source_feature_name(source_render_pass)
              source_feature = source_features[Naming.source_feature_name(source_render_pass, index=source_index)]
              self._first_statistics_iteration(source_feature, source_render_pass, source_feature_name, False, target_features)
              if (
                  RenderPasses.is_direct_or_indirect_render_pass(source_render_pass) and not
                  RenderPasses.is_volume_render_pass(source_render_pass)):
                # TODO: Make sure the required target feature is present!
                source_feature_name = Naming.source_feature_name(source_render_pass, masked=True)
                self._first_statistics_iteration(source_feature, source_render_pass, source_feature_name, True, target_features)
              
          for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
            target_feature_name = Naming.target_feature_name(target_render_pass)
            target_feature = target_features[target_feature_name]
            self._first_statistics_iteration(source_feature, target_render_pass, target_feature_name, False, target_features)
            if (
                RenderPasses.is_direct_or_indirect_render_pass(target_render_pass) and not
                RenderPasses.is_volume_render_pass(target_render_pass)):
              target_feature_name = Naming.target_feature_name(target_render_pass, masked=True)
              self._first_statistics_iteration(target_feature, target_render_pass, target_feature_name, True, target_features)
            
        except tf.errors.OutOfRangeError:
          break
      
      
      # The arrays of values need to be joined to get one number.
      
      for feature_name in self.minimums:
        self.minimums[feature_name] = self.minimums[feature_name].numpy().item()
        self.maximums[feature_name] = self.maximums[feature_name].numpy().item()
        mean = self.means[feature_name]
        self.means[feature_name] = tf.reduce_mean(mean).numpy().item()
        
        self.minimums_log1p[feature_name] = self.minimums_log1p[feature_name].numpy().item()
        self.maximums_log1p[feature_name] = self.maximums_log1p[feature_name].numpy().item()
        mean_log1p = self.means_log1p[feature_name]
        self.means_log1p[feature_name] = tf.reduce_mean(mean_log1p).numpy().item()
      
      
      # Iterate again through all the tfrecords to compute the variance, based on the mean.
      
      iterator = self._dataset_iterator(samples_per_pixel)
      while True:
        try:
          source_features, target_features = iterator.get_next()
          
          for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
            for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
              source_feature_name = Naming.source_feature_name(source_render_pass)
              source_feature = source_features[Naming.source_feature_name(source_render_pass, index=source_index)]
              self._second_statistics_iteration(source_feature, source_render_pass, source_feature_name, False, target_features)
              if (
                  RenderPasses.is_direct_or_indirect_render_pass(source_render_pass) and not
                  RenderPasses.is_volume_render_pass(source_render_pass)):
                source_feature_name = Naming.source_feature_name(source_render_pass, masked=True)
                self._second_statistics_iteration(source_feature, source_render_pass, source_feature_name, True, target_features)
              
          for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
            target_feature_name = Naming.target_feature_name(target_render_pass)
            target_feature = target_features[target_feature_name]
            self._second_statistics_iteration(source_feature, target_render_pass, target_feature_name, False, target_features)
            if (
                RenderPasses.is_direct_or_indirect_render_pass(target_render_pass) and not
                RenderPasses.is_volume_render_pass(target_render_pass)):
              target_feature_name = Naming.target_feature_name(target_render_pass, masked=True)
              self._second_statistics_iteration(target_feature, target_render_pass, target_feature_name, True, target_features)
          
        except tf.errors.OutOfRangeError:
          break
      
      
      # Join the results again.
      
      for feature_name in self.variances:
        variance = self.variances[feature_name]
        self.variances[feature_name] = tf.reduce_mean(variance).numpy().item()
        
        variance_log1p = self.variances_log1p[feature_name]
        self.variances_log1p[feature_name] = tf.reduce_mean(variance_log1p).numpy().item()
      
      
      # Integrate the results into statistics.
      
      for feature_name in self.minimums:
        
        # REMARK: The 'current_' prefix is only used to avoid a name clash.
        current_statistics = Statistics(
            self.minimums[feature_name], self.maximums[feature_name],
            self.means[feature_name], self.variances[feature_name])
        current_statistics_log1p = Statistics(
            self.minimums_log1p[feature_name], self.maximums_log1p[feature_name],
            self.means_log1p[feature_name], self.variances_log1p[feature_name])
        feature_statistics = FeatureStatistics(
            RenderPasses.number_of_channels(feature_name.split('/')[-1]), current_statistics, current_statistics_log1p)
        statistics[feature_name] = feature_statistics
      
      
      # Save the statistics.
      
      filename = self.tfrecords_creator.name + '.json'
      if self.tfrecords_creator.group_by_samples_per_pixel:
        filename = self.tfrecords_creator.name + '_' + str(samples_per_pixel) + '.json'

      statistics_json_filename = os.path.join(self.tfrecords_creator.base_tfrecords_directory, filename)
      statistics_json_content = json.dumps(statistics, cls=DataStatisticsEncoder, sort_keys=True, indent=2)
      with open(statistics_json_filename, 'w+', encoding='utf-8') as statistics_json_file:
        statistics_json_file.write(statistics_json_content)

      
  def _first_statistics_iteration(self, feature, render_pass_name, feature_name, use_mask, target_features):
    
    feature_log1p = Utilities.signed_log1p(feature)
    
    if use_mask:
      # For direct and indirect passes, we only care about the relevant pixels. We create a mask for this.
      # It depends on the corresponding ground truth color pass. Whenever that one is not black, the pixels
      # of the direct and indirect passes matter.
      
      corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(render_pass_name)
      corresponding_target_feature = target_features[Naming.target_feature_name(corresponding_color_pass)]
      mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
      mask_sum = tf.reduce_sum(mask)
      
      # Adjust the mask, such that it can be multiplied with the feature.
      mask = tf.stack([mask, mask, mask], axis=2)
      
      
      self.minimums[feature_name] = tf.minimum(self.minimums[feature_name], tf.reduce_min(feature))
      self.maximums[feature_name] = tf.maximum(self.maximums[feature_name], tf.reduce_max(feature))
      if tf.greater(mask_sum, 0.):
        self.means[feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(feature, mask), mask_sum)))
      
      self.minimums_log1p[feature_name] = tf.minimum(self.minimums_log1p[feature_name], tf.reduce_min(feature_log1p))
      self.maximums_log1p[feature_name] = tf.maximum(self.maximums_log1p[feature_name], tf.reduce_max(feature_log1p))
      if tf.greater(mask_sum, 0.):
        self.means_log1p[feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(feature_log1p, mask), mask_sum)))
      
    else:
      self.minimums[feature_name] = tf.minimum(self.minimums[feature_name], tf.reduce_min(feature))
      self.maximums[feature_name] = tf.maximum(self.maximums[feature_name], tf.reduce_max(feature))
      self.means[feature_name].append(tf.reduce_mean(feature))
      
      self.minimums_log1p[feature_name] = tf.minimum(self.minimums_log1p[feature_name], tf.reduce_min(feature_log1p))
      self.maximums_log1p[feature_name] = tf.maximum(self.maximums_log1p[feature_name], tf.reduce_max(feature_log1p))
      self.means_log1p[feature_name].append(tf.reduce_mean(feature_log1p))
  
  def _second_statistics_iteration(self, feature, render_pass_name, feature_name, use_mask, target_features):
    
    mean = self.means[feature_name]
    mean_log1p = self.means_log1p[feature_name]
    feature_log1p = Utilities.signed_log1p(feature)
    
    if use_mask:
      # For direct and indirect passes, we only care about the relevant pixels. We create a mask for this.
      # It depends on the corresponding ground truth color pass. Whenever that one is not black, the pixels
      # of the direct and indirect passes matter.
      
      corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(render_pass_name)
      corresponding_target_feature = target_features[Naming.target_feature_name(corresponding_color_pass)]
      mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
      mask_sum = tf.reduce_sum(mask)
      
      # Adjust the mask, such that it can be multiplied with the feature.
      mask = tf.stack([mask, mask, mask], axis=2)
      
      
      if tf.greater(mask_sum, 0.):
        self.variances[feature_name].append(
            tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(feature, mean)), mask), mask_sum)))
      if tf.greater(mask_sum, 0.):
        self.variances_log1p[feature_name].append(
            tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(feature_log1p, mean_log1p)), mask), mask_sum)))
      
    else:
      self.variances[feature_name].append(tf.reduce_mean(tf.square(tf.subtract(feature, mean))))
      self.variances_log1p[feature_name].append(tf.reduce_mean(tf.square(tf.subtract(feature_log1p, mean_log1p))))
    
  def _dataset_iterator(self, source_samples_per_pixel=None):
    directory = os.path.join(self.tfrecords_creator.base_tfrecords_directory, self.tfrecords_creator.name)
    if source_samples_per_pixel != None:
      directory = os.path.join(directory, str(source_samples_per_pixel))
    files = tf.data.Dataset.list_files(directory + '/*')
    
    threads = multiprocessing.cpu_count()
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', buffer_size=None, num_parallel_reads=threads)
    dataset = dataset.map(map_func=self._feature_parser, num_parallel_calls=threads)
    iterator = tfe.Iterator(dataset)
    return iterator

  def _feature_parser(self, serialized_example):
    features = {}
    for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
      for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
        features[Naming.source_feature_name(source_render_pass, index=source_index)] = tf.FixedLenFeature([], tf.string)
    for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
      features[Naming.target_feature_name(target_render_pass)] = tf.FixedLenFeature([], tf.string)
    
    parsed_features = tf.parse_single_example(serialized_example, features)
    
    source_features = {}
    for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
      for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
        source_feature = tf.decode_raw(
            parsed_features[Naming.source_feature_name(source_render_pass, index=source_index)], tf.float32)
        number_of_channels = RenderPasses.number_of_channels(source_render_pass)
        source_feature = tf.reshape(
            source_feature, [self.tfrecords_creator.tiles_height_width, self.tfrecords_creator.tiles_height_width, number_of_channels])
        source_features[Naming.source_feature_name(source_render_pass, index=source_index)] = source_feature
    
    target_features = {}
    for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
      target_feature = tf.decode_raw(
          parsed_features[Naming.target_feature_name(target_render_pass)], tf.float32)
      number_of_channels = RenderPasses.number_of_channels(target_render_pass)
      target_feature = tf.reshape(
          target_feature, [self.tfrecords_creator.tiles_height_width, self.tfrecords_creator.tiles_height_width, number_of_channels])
      target_features[Naming.target_feature_name(target_render_pass)] = target_feature
    
    return source_features, target_features


class DataStatisticsEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, '__json__'):
      return obj.__json__()
    if hasattr(obj, '__dict__'):
      return obj.__dict__
    return json.JSONEncoder.default(self, obj)