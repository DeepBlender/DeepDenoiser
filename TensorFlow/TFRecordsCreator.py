from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import multiprocessing

import json
import gzip

from RenderPasses import RenderPasses
from RenderPasses import RenderPassesUsage
from FeatureStatistics import Statistics
from FeatureStatistics import FeatureStatistics
from RenderDirectories import RenderDirectories
import Utilities

parser = argparse.ArgumentParser(description='Create tfrecords files for the DeepDenoiser.')
parser.add_argument('json_filename', help='The json specifying all the relevant details.')


class TFRecordsCreator:

  # TODO: Add log file functionality from previous version. (DeepBlender)

  def __init__(
      self, name, base_tfrecords_directory, base_render_directory, relative_render_directories,
      source_samples_per_pixel, source_render_passes_usage, number_of_sources_per_example,
      target_samples_per_pixel, target_render_passes_usage,
      tiles_height_width, tiles_per_tfrecords):
    self.name = name
    self.base_tfrecords_directory = base_tfrecords_directory
    self.source_samples_per_pixel = source_samples_per_pixel
    self.source_render_passes_usage = source_render_passes_usage
    self.number_of_sources_per_example = number_of_sources_per_example
    self.target_samples_per_pixel = target_samples_per_pixel
    self.target_render_passes_usage = target_render_passes_usage
    self.tiles_height_width = tiles_height_width
    self.tiles_per_tfrecords = tiles_per_tfrecords
    
    self.render_directories_list = []
    for render_directories in relative_render_directories:
      new_render_directories = RenderDirectories(os.path.join(base_render_directory, render_directories))
      
      # TODO: Certainly not the best way to perform the validity checks. (DeepBlender)
      assert new_render_directories.required_files_exist(self.source_samples_per_pixel, self.source_render_passes_usage)
      assert self.number_of_sources_per_example == len(new_render_directories.samples_per_pixel_to_render_directories[self.source_samples_per_pixel])
      
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel is 'best':
        target_samples_per_pixel = new_render_directories.ground_truth_samples_per_pixel()
      assert new_render_directories.required_files_exist(target_samples_per_pixel, self.target_render_passes_usage)
      
      self.render_directories_list.append(new_render_directories)
  
  def create_tfrecords(self):
    tfrecords_writer = TFRecordsWriter(self.name, self.base_tfrecords_directory, self.tiles_per_tfrecords)
  
    for render_directories in self.render_directories_list:
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel == 'best':
        target_samples_per_pixel = render_directories.ground_truth_samples_per_pixel()
      
      render_directories.load_images(self.source_samples_per_pixel, self.source_render_passes_usage)
      render_directories.load_images(target_samples_per_pixel, self.target_render_passes_usage)
      
      # TODO: Find a better way to deal with it! (DeepBlender)
      assert render_directories.have_loaded_images_identical_sizes()
      
      
      height, width = render_directories.size_of_loaded_images()
      
      # Split the images into tiles
      tiles_x_count = height // self.tiles_height_width
      tiles_y_count = width // self.tiles_height_width
      
      for i in range(tiles_x_count):
        for j in range (tiles_y_count):
          x1 = i * self.tiles_height_width
          x2 = (i + 1) * self.tiles_height_width
          y1 = j * self.tiles_height_width
          y2 = (j + 1) * self.tiles_height_width
          
          features = {}
          
          # Prepare the source image tile.
          source_index = 0
          for source_render_directory in render_directories.samples_per_pixel_to_render_directories[self.source_samples_per_pixel]:
            for source_render_pass in source_render_directory.render_pass_to_image:
              image = source_render_directory.render_pass_to_image[source_render_pass]
              features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)] = TFRecordsCreator._bytes_feature(tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
            source_index = source_index + 1
      
          # Prepare the target image tiles.
          target_render_directory = render_directories.samples_per_pixel_to_render_directories[target_samples_per_pixel][0]
          for target_render_pass in target_render_directory.render_pass_to_image:
            image = target_render_directory.render_pass_to_image[target_render_pass]
            features[RenderPasses.target_feature_name(target_render_pass)] = TFRecordsCreator._bytes_feature(tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
          
          tfrecords_writer.write(features)
      
      render_directories.unload_images()
    tfrecords_writer.close()
  
  
  def create_statistics(self):
    statistics = {}
    statistics['tiles_height_width'] = self.tiles_height_width
    statistics['number_of_sources_per_example'] = self.number_of_sources_per_example
    statistics['source_samples_per_pixel'] = self.source_samples_per_pixel
    
    statistics = self._complete_statistics(statistics)
    
    statistics_json_filename = os.path.join(self.base_tfrecords_directory, self.name + '.json')
    statistics_json_content = json.dumps(statistics, cls=DataStatisticsEncoder, sort_keys=True, indent=2)
    with open(statistics_json_filename, 'w+', encoding='utf-8') as statistics_json_file:
      statistics_json_file.write(statistics_json_content)
  
  def _complete_statistics(self, statistics):
    minimums = {}
    maximums = {}
    means = {}
    variances = {}
    minimums_log1p = {}
    maximums_log1p = {}
    means_log1p = {}
    variances_log1p = {}
    
    for source_render_pass in self.source_render_passes_usage.render_passes():
      source_feature_name = RenderPasses.source_feature_name(source_render_pass)
      minimums[source_feature_name] = []
      maximums[source_feature_name] = []
      means[source_feature_name] = []
      variances[source_feature_name] = []
      minimums_log1p[source_feature_name] = []
      maximums_log1p[source_feature_name] = []
      means_log1p[source_feature_name] = []
      variances_log1p[source_feature_name] = []
    
    for target_render_pass in self.target_render_passes_usage.render_passes():
      target_feature_name = RenderPasses.target_feature_name(target_render_pass)
      minimums[target_feature_name] = []
      maximums[target_feature_name] = []
      means[target_feature_name] = []
      variances[target_feature_name] = []
      minimums_log1p[target_feature_name] = []
      maximums_log1p[target_feature_name] = []
      means_log1p[target_feature_name] = []
      variances_log1p[target_feature_name] = []
    
    iterator = self._dataset_iterator()
    while True:
      try:
        source_features, target_features = iterator.get_next()
        
        for source_index in range(self.number_of_sources_per_example):
          for source_render_pass in self.source_render_passes_usage.render_passes():
            source_feature_name = RenderPasses.source_feature_name(source_render_pass)
            
            source_feature = source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)]
            minimums[source_feature_name].append(tf.reduce_min(source_feature))
            maximums[source_feature_name].append(tf.reduce_max(source_feature))
            means[source_feature_name].append(tf.reduce_mean(source_feature))
            
            source_feature_log1p = Utilities.signed_log1p(source_feature)
            minimums_log1p[source_feature_name].append(tf.reduce_min(source_feature_log1p))
            maximums_log1p[source_feature_name].append(tf.reduce_max(source_feature_log1p))
            means_log1p[source_feature_name].append(tf.reduce_mean(source_feature_log1p))
            
        for target_render_pass in self.target_render_passes_usage.render_passes():
          target_feature_name = RenderPasses.target_feature_name(target_render_pass)
          
          target_feature = target_features[RenderPasses.target_feature_name(target_render_pass)]
          minimums[target_feature_name].append(tf.reduce_min(target_feature))
          maximums[target_feature_name].append(tf.reduce_max(target_feature))
          means[target_feature_name].append(tf.reduce_mean(target_feature))
          
          target_feature_log1p = Utilities.signed_log1p(target_feature)
          minimums_log1p[target_feature_name].append(tf.reduce_min(target_feature_log1p))
          maximums_log1p[target_feature_name].append(tf.reduce_max(target_feature_log1p))
          means_log1p[target_feature_name].append(tf.reduce_mean(target_feature_log1p))
          
      except tf.errors.OutOfRangeError:
        break
    
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
    
    iterator = self._dataset_iterator()
    while True:
      try:
        source_features, target_features = iterator.get_next()
        
        for source_index in range(self.number_of_sources_per_example):
          for source_render_pass in self.source_render_passes_usage.render_passes():
            source_feature_name = RenderPasses.source_feature_name(source_render_pass)
            
            mean = means[source_feature_name]
            source_feature = source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)]
            variances[source_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(source_feature, mean))))
            
            mean_log1p = means_log1p[source_feature_name]
            source_feature_log1p = Utilities.signed_log1p(source_feature)
            variances_log1p[source_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(source_feature_log1p, mean_log1p))))
            
        for target_render_pass in self.target_render_passes_usage.render_passes():
          target_feature_name = RenderPasses.target_feature_name(target_render_pass)
          
          mean = means[target_feature_name]
          target_feature = target_features[RenderPasses.target_feature_name(target_render_pass)]
          variances[target_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(target_feature, mean))))
          
          mean_log1p = means_log1p[target_feature_name]
          target_feature_log1p = Utilities.signed_log1p(target_feature)
          variances_log1p[target_feature_name].append(tf.reduce_mean(tf.square(tf.subtract(target_feature_log1p, mean_log1p))))
        
      except tf.errors.OutOfRangeError:
        break
    
    for feature_name in variances:
      variance = variances[feature_name]
      variances[feature_name] = tf.reduce_mean(variance).numpy().item()
      
      variance_log1p = variances_log1p[feature_name]
      variances_log1p[feature_name] = tf.reduce_min(variance_log1p).numpy().item()
    
    
    # Integrate the results into statistics
    for feature_name in minimums:
      
      # REMARK: The 'current_' prefix is only used to avoid a name clash.
      current_statistics = Statistics(minimums[feature_name], maximums[feature_name], means[feature_name], variances[feature_name])
      current_statistics_log1p = Statistics(minimums_log1p[feature_name], maximums_log1p[feature_name], means_log1p[feature_name], variances_log1p[feature_name])
      feature_statistics = FeatureStatistics(RenderPasses.number_of_channels(feature_name.split('/')[-1]), current_statistics, current_statistics_log1p)
      statistics[feature_name] = feature_statistics
    
    return statistics
  
  def _dataset_iterator(self):
    directory = os.path.join(self.base_tfrecords_directory, self.name)
    files = tf.data.Dataset.list_files(directory + '/*', shuffle=True)
    
    threads = multiprocessing.cpu_count()
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', buffer_size=None, num_parallel_reads=threads)
    dataset = dataset.map(map_func=self._feature_parser, num_parallel_calls=threads)
    iterator = tfe.Iterator(dataset)
    return iterator
  
  def _feature_parser(self, serialized_example):
      features = {}
      for source_index in range(self.number_of_sources_per_example):
        for source_render_pass in self.source_render_passes_usage.render_passes():
          features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)] = tf.FixedLenFeature([], tf.string)
      for target_render_pass in self.target_render_passes_usage.render_passes():
        features[RenderPasses.target_feature_name(target_render_pass)] = tf.FixedLenFeature([], tf.string)
      
      parsed_features = tf.parse_single_example(serialized_example, features)
      
      source_features = {}
      for source_index in range(self.number_of_sources_per_example):
        for source_render_pass in self.source_render_passes_usage.render_passes():
          source_feature = tf.decode_raw(parsed_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)], tf.float32)
          number_of_channels = RenderPasses.number_of_channels(source_render_pass)
          source_feature = tf.reshape(source_feature, [self.tiles_height_width, self.tiles_height_width, number_of_channels])
          source_features[RenderPasses.source_feature_name_indexed(source_render_pass, source_index)] = source_feature
      
      target_features = {}
      for target_render_pass in self.target_render_passes_usage.render_passes():
        target_feature = tf.decode_raw(parsed_features[RenderPasses.target_feature_name(target_render_pass)], tf.float32)
        number_of_channels = RenderPasses.number_of_channels(target_render_pass)
        target_feature = tf.reshape(target_feature, [self.tiles_height_width, self.tiles_height_width, number_of_channels])
        target_features[RenderPasses.target_feature_name(target_render_pass)] = target_feature
      
      return source_features, target_features
  
  
  def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

  def _float_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class TFRecordsWriter:
  def __init__(self, name, base_directory, tiles_per_tfrecords):
    self.name = name
    self.base_directory = base_directory
    self.tiles_per_tfrecords = tiles_per_tfrecords
    self.tfrecords_directory = os.path.join(self.base_directory, self.name)
    if not os.path.exists(self.tfrecords_directory):
      os.makedirs(self.tfrecords_directory)
    self.writer = None
    self.tfrecords_index = 0
    self.added_tiles = 0
  
  def write(self, features):
    if self.writer == None:
      self.tfrecords_filename = os.path.join(self.tfrecords_directory, self.name + '_' + str(self.tfrecords_index) + '.tfrecords')
      self.writer = tf.python_io.TFRecordWriter(self.tfrecords_filename)
    
    example = tf.train.Example(features=tf.train.Features(feature=features))
    self.writer.write(example.SerializeToString())
    self.added_tiles = self.added_tiles + 1
    
    if self.added_tiles >= self.tiles_per_tfrecords:
      self.close()
  
  def close(self):
    if self.writer != None:
      self.writer.close()
      TFRecordsWriter._compress(self.tfrecords_filename)
    self.added_tiles = 0
    self.tfrecords_filename = None
    self.writer = None
    self.tfrecords_index = self.tfrecords_index + 1
  
  def _compress(filename, delete_uncompressed=True):
    gzip_filename = filename + '.gz'
    original_file = open(filename, 'rb')
    gzip_file = gzip.open(gzip_filename, 'wb')
    gzip_file.writelines(original_file)
    gzip_file.close()
    original_file.close()
    if delete_uncompressed:
      os.remove(filename)


class DataStatisticsEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, '__json__'):
      return obj.__json__()
    if hasattr(obj, '__dict__'):
      return obj.__dict__
    return json.JSONEncoder.default(self, obj)


def main(parsed_arguments):
  try:
    json_filename = parsed_arguments.json_filename
    json_content = open(json_filename, 'r').read()
    parsed_json = json.loads(json_content)
  except:
    print('Expected a valid json file as argument.')
  
  base_render_directory = parsed_json['base_render_directory']
  base_tfrecords_directory = parsed_json['base_tfrecords_directory']
  mode_name_to_mode_settings = parsed_json['modes']
  
  source = parsed_json['source']
  source_samples_per_pixel = source['samples_per_pixel']
  number_of_sources_per_example = source['number_of_sources_per_example']
  source_render_passes_usage = RenderPassesUsage()
  source_render_passes_usage.__dict__ = source['features']
  
  target = parsed_json['target']
  target_samples_per_pixel = target['samples_per_pixel']
  target_render_passes_usage = RenderPassesUsage()
  target_render_passes_usage.__dict__ = target['features']
  
  tfrecords_creators = []
  for mode_name in mode_name_to_mode_settings:
    mode_settings = mode_name_to_mode_settings[mode_name]
    tfrecords_creator = TFRecordsCreator(
        mode_name, base_tfrecords_directory, base_render_directory, mode_settings['render_directories'],
        source_samples_per_pixel, source_render_passes_usage, number_of_sources_per_example,
        target_samples_per_pixel, target_render_passes_usage,
        mode_settings['tiles_height_width'], mode_settings['tiles_per_tfrecords'])
    tfrecords_creators.append(tfrecords_creator)
    
  for tfrecords_creator in tfrecords_creators:
    tfrecords_creator.create_tfrecords()
  
  tf.enable_eager_execution()
  for tfrecords_creator in tfrecords_creators:
    tfrecords_creator.create_statistics()
  
if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)