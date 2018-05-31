import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import multiprocessing

import json

from RenderPasses import RenderPasses
from RenderPasses import RenderPassesUsage
from FeatureStatistics import Statistics
from FeatureStatistics import FeatureStatistics
import Utilities
import Conv2dUtilities


class TFRecordsStatistics:
  def __init__(self, tfrecords_creator):
    self.tfrecords_creator = tfrecords_creator
  
  def compute_and_save_statistics(self):
    statistics = {}
    statistics['tiles_height_width'] = self.tfrecords_creator.tiles_height_width
    statistics['number_of_sources_per_example'] = self.tfrecords_creator.number_of_sources_per_example
    statistics['source_samples_per_pixel'] = self.tfrecords_creator.source_samples_per_pixel
  
  
    # Prepare the data structures.
  
    minimums = {}
    maximums = {}
    means = {}
    variances = {}
    minimums_log1p = {}
    maximums_log1p = {}
    means_log1p = {}
    variances_log1p = {}
    
    for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
      source_feature_name = RenderPasses.source_feature_name(source_render_pass)
      minimums[source_feature_name] = []
      maximums[source_feature_name] = []
      means[source_feature_name] = []
      variances[source_feature_name] = []
      minimums_log1p[source_feature_name] = []
      maximums_log1p[source_feature_name] = []
      means_log1p[source_feature_name] = []
      variances_log1p[source_feature_name] = []
    
    for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
      target_feature_name = RenderPasses.target_feature_name(target_render_pass)
      minimums[target_feature_name] = []
      maximums[target_feature_name] = []
      means[target_feature_name] = []
      variances[target_feature_name] = []
      minimums_log1p[target_feature_name] = []
      maximums_log1p[target_feature_name] = []
      means_log1p[target_feature_name] = []
      variances_log1p[target_feature_name] = []
    
    
    # Iterate through the tfrecords to compute the usual and log1p statistics for minimum, maximum and mean.
    
    iterator = self._dataset_iterator()
    while True:
      try:
        source_features, target_features = iterator.get_next()
        
        for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
        
          for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
            source_feature_name = RenderPasses.source_feature_name(source_render_pass)
            
            # For direct and indirect passes, we only care about the relevant pixels. We create a mask for this.
            # It depends on the corresponding ground truth color pass. Whenever that one is not black, the pixels
            # of the direct and indirect passes matter.
            
            use_mask = False
            if RenderPasses.is_direct_or_indirect_render_pass(source_render_pass):
              corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(source_render_pass)
              corresponding_target_feature = target_features[RenderPasses.target_feature_name(corresponding_color_pass)]
              mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
              mask_sum = tf.reduce_sum(mask)
              use_mask = True
              
              # Adjust the mask, such that it can be multiplied with the feature.
              mask = tf.stack([mask, mask, mask], axis=2)
            
            source_feature = source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)]
            minimums[source_feature_name].append(tf.reduce_min(source_feature))
            maximums[source_feature_name].append(tf.reduce_max(source_feature))
            if use_mask:
              if tf.greater(mask_sum, 0.):
                means[source_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(source_feature, mask), mask_sum)))
            else:
              means[source_feature_name].append(tf.reduce_mean(source_feature))
            
            source_feature_log1p = Utilities.signed_log1p(source_feature)
            minimums_log1p[source_feature_name].append(tf.reduce_min(source_feature_log1p))
            maximums_log1p[source_feature_name].append(tf.reduce_max(source_feature_log1p))
            if use_mask:
              if tf.greater(mask_sum, 0.):
                means_log1p[source_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(source_feature_log1p, mask), mask_sum)))
            else:
              means_log1p[source_feature_name].append(tf.reduce_mean(source_feature_log1p))
            
        for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
          target_feature_name = RenderPasses.target_feature_name(target_render_pass)
          
          use_mask = False
          if RenderPasses.is_direct_or_indirect_render_pass(target_render_pass):
            corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(target_render_pass)
            corresponding_target_feature = target_features[RenderPasses.target_feature_name(corresponding_color_pass)]
            mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
            mask_sum = tf.reduce_sum(mask)
            use_mask = True
            
            # Adjust the mask, such that it can be multiplied with the feature.
            mask = tf.stack([mask, mask, mask], axis=2)
          
          target_feature = target_features[RenderPasses.target_feature_name(target_render_pass)]
          minimums[target_feature_name].append(tf.reduce_min(target_feature))
          maximums[target_feature_name].append(tf.reduce_max(target_feature))
          if use_mask:
            if tf.greater(mask_sum, 0.):
              means[target_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(target_feature, mask), mask_sum)))
          else:
            means[target_feature_name].append(tf.reduce_mean(target_feature))
          
          target_feature_log1p = Utilities.signed_log1p(target_feature)
          minimums_log1p[target_feature_name].append(tf.reduce_min(target_feature_log1p))
          maximums_log1p[target_feature_name].append(tf.reduce_max(target_feature_log1p))
          if use_mask:
            if tf.greater(mask_sum, 0.):
              means_log1p[target_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(target_feature_log1p, mask), mask_sum)))
          else:
            means_log1p[target_feature_name].append(tf.reduce_mean(target_feature_log1p))
          
      except tf.errors.OutOfRangeError:
        break
    
    
    # The arrays of values need to be joined to get one number.
    
    for feature_name in minimums:
      minimum = minimums[feature_name]
      minimums[feature_name] = tf.reduce_min(minimum).numpy().item()
      maximum = maximums[feature_name]
      maximums[feature_name] = tf.reduce_max(maximum).numpy().item()
      mean = means[feature_name]
      means[feature_name] = tf.reduce_mean(mean).numpy().item()
      
      minimum_log1p = minimums_log1p[feature_name]
      minimums_log1p[feature_name] = tf.reduce_min(minimum_log1p).numpy().item()
      maximum_log1p = maximums_log1p[feature_name]
      maximums_log1p[feature_name] = tf.reduce_max(maximum_log1p).numpy().item()
      mean_log1p = means_log1p[feature_name]
      means_log1p[feature_name] = tf.reduce_mean(mean_log1p).numpy().item()
    
    
    # Iterate again through all the tfrecords to compute the variance, based on the mean.
    
    iterator = self._dataset_iterator()
    while True:
      try:
        source_features, target_features = iterator.get_next()
        
        for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
          for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
            source_feature_name = RenderPasses.source_feature_name(source_render_pass)
            
            use_mask = False
            if RenderPasses.is_direct_or_indirect_render_pass(source_render_pass):
              corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(source_render_pass)
              corresponding_target_feature = target_features[RenderPasses.target_feature_name(corresponding_color_pass)]
              mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
              mask_sum = tf.reduce_sum(mask)
              use_mask = True
              
              # Adjust the mask, such that it can be multiplied with the feature.
              mask = tf.stack([mask, mask, mask], axis=2)
            
            mean = means[source_feature_name]
            source_feature = source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)]
            if use_mask:
              if tf.greater(mask_sum, 0.):
                variances[source_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(source_feature, mean)), mask), mask_sum)))
            else:
              variances[source_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(source_feature, mean))))
            
            mean_log1p = means_log1p[source_feature_name]
            source_feature_log1p = Utilities.signed_log1p(source_feature)
            if use_mask:
              if tf.greater(mask_sum, 0.):
                variances_log1p[source_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(source_feature_log1p, mean_log1p)), mask), mask_sum)))
            else:
              variances_log1p[source_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(source_feature_log1p, mean_log1p))))
            
        for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
          target_feature_name = RenderPasses.target_feature_name(target_render_pass)
          
          use_mask = False
          if RenderPasses.is_direct_or_indirect_render_pass(target_render_pass):
            corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(target_render_pass)
            corresponding_target_feature = target_features[RenderPasses.target_feature_name(corresponding_color_pass)]
            mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
            mask_sum = tf.reduce_sum(mask)
            use_mask = True
            
            # Adjust the mask, such that it can be multiplied with the feature.
            mask = tf.stack([mask, mask, mask], axis=2)
          
          mean = means[target_feature_name]
          target_feature = target_features[RenderPasses.target_feature_name(target_render_pass)]
          if use_mask:
            if tf.greater(mask_sum, 0.):
              variances[target_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(target_feature, mean)), mask), mask_sum)))
          else:
            variances[target_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(target_feature, mean))))
          
          mean_log1p = means_log1p[target_feature_name]
          target_feature_log1p = Utilities.signed_log1p(target_feature)
          if use_mask:
            if tf.greater(mask_sum, 0.):
              variances_log1p[target_feature_name].append(tf.reduce_sum(tf.divide(tf.multiply(tf.square(tf.subtract(target_feature_log1p, mean_log1p)), mask), mask_sum)))
          else:
            variances_log1p[target_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(target_feature_log1p, mean_log1p))))
        
      except tf.errors.OutOfRangeError:
        break
    
    
    # Join the results again.
    
    for feature_name in variances:
      variance = variances[feature_name]
      variances[feature_name] = tf.reduce_mean(variance).numpy().item()
      
      variance_log1p = variances_log1p[feature_name]
      variances_log1p[feature_name] = tf.reduce_min(variance_log1p).numpy().item()
    
    
    # Integrate the results into statistics.
    
    for feature_name in minimums:
      
      # REMARK: The 'current_' prefix is only used to avoid a name clash.
      current_statistics = Statistics(minimums[feature_name], maximums[feature_name], means[feature_name], variances[feature_name])
      current_statistics_log1p = Statistics(minimums_log1p[feature_name], maximums_log1p[feature_name], means_log1p[feature_name], variances_log1p[feature_name])
      feature_statistics = FeatureStatistics(RenderPasses.number_of_channels(feature_name.split('/')[-1]), current_statistics, current_statistics_log1p)
      statistics[feature_name] = feature_statistics
    
    
    # Save the statistics.
    
    statistics_json_filename = os.path.join(self.tfrecords_creator.base_tfrecords_directory, self.tfrecords_creator.name + '.json')
    statistics_json_content = json.dumps(statistics, cls=DataStatisticsEncoder, sort_keys=True, indent=2)
    with open(statistics_json_filename, 'w+', encoding='utf-8') as statistics_json_file:
      statistics_json_file.write(statistics_json_content)
  
  def _dataset_iterator(self):
    directory = os.path.join(self.tfrecords_creator.base_tfrecords_directory, self.tfrecords_creator.name)
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
        features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)] = tf.FixedLenFeature([], tf.string)
    for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
      features[RenderPasses.target_feature_name(target_render_pass)] = tf.FixedLenFeature([], tf.string)
    
    parsed_features = tf.parse_single_example(serialized_example, features)
    
    source_features = {}
    for source_index in range(self.tfrecords_creator.number_of_sources_per_example):
      for source_render_pass in self.tfrecords_creator.source_render_passes_usage.render_passes():
        source_feature = tf.decode_raw(parsed_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)], tf.float32)
        number_of_channels = RenderPasses.number_of_channels(source_render_pass)
        source_feature = tf.reshape(source_feature, [self.tfrecords_creator.tiles_height_width, self.tfrecords_creator.tiles_height_width, number_of_channels])
        source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)] = source_feature
    
    target_features = {}
    for target_render_pass in self.tfrecords_creator.target_render_passes_usage.render_passes():
      target_feature = tf.decode_raw(parsed_features[RenderPasses.target_feature_name(target_render_pass)], tf.float32)
      number_of_channels = RenderPasses.number_of_channels(target_render_pass)
      target_feature = tf.reshape(target_feature, [self.tfrecords_creator.tiles_height_width, self.tfrecords_creator.tiles_height_width, number_of_channels])
      target_features[RenderPasses.target_feature_name(target_render_pass)] = target_feature
    
    return source_features, target_features

class DataStatisticsEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, '__json__'):
      return obj.__json__()
    if hasattr(obj, '__dict__'):
      return obj.__dict__
    return json.JSONEncoder.default(self, obj)