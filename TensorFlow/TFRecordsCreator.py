from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import json
import gzip

from Naming import Naming
from RenderPasses import RenderPassesUsage
from TFRecordsStatistics import TFRecordsStatistics
from RenderDirectories import RenderDirectories


parser = argparse.ArgumentParser(description='Create tfrecords files for the DeepDenoiser.')

parser.add_argument(
    'json_filename',
    help='The json specifying all the relevant details.')

parser.add_argument(
    '--statistics', action="store_true",
    help='Only recalculate the statistics.')

class TFRecordsCreator:

  # TODO: Add log file functionality from previous version. (DeepBlender)

  def __init__(
      self, name, base_tfrecords_directory, base_render_directory, relative_render_directories,
      source_samples_per_pixel, source_render_passes_usage, number_of_sources_per_example,
      target_samples_per_pixel, target_render_passes_usage,
      tiles_height_width, examples_per_tfrecords):
    self.name = name
    self.base_tfrecords_directory = base_tfrecords_directory
    self.source_samples_per_pixel = source_samples_per_pixel
    self.source_render_passes_usage = source_render_passes_usage
    self.number_of_sources_per_example = number_of_sources_per_example
    self.target_samples_per_pixel = target_samples_per_pixel
    self.target_render_passes_usage = target_render_passes_usage
    self.tiles_height_width = tiles_height_width
    self.examples_per_tfrecords = examples_per_tfrecords
    
    self.render_directories_list = []
    for render_directories in relative_render_directories:
      new_render_directories = RenderDirectories(os.path.join(base_render_directory, render_directories))
      
      # TODO: Certainly not the best way to perform the validity checks. (DeepBlender)
      assert new_render_directories.required_files_exist(self.source_samples_per_pixel, self.source_render_passes_usage)
      assert (
          self.number_of_sources_per_example <= len(
              new_render_directories.samples_per_pixel_to_render_directories[self.source_samples_per_pixel]))
      
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel is 'best':
        target_samples_per_pixel = new_render_directories.ground_truth_samples_per_pixel()
      assert new_render_directories.required_files_exist(target_samples_per_pixel, self.target_render_passes_usage)
      
      self.render_directories_list.append(new_render_directories)
  
  def create_tfrecords(self):
    tfrecords_writer = TFRecordsWriter(self.name, self.base_tfrecords_directory, self.examples_per_tfrecords)
  
    for render_directories in self.render_directories_list:
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel == 'best':
        target_samples_per_pixel = render_directories.ground_truth_samples_per_pixel()
      
      render_directories.load_images(self.source_samples_per_pixel, self.source_render_passes_usage)
      render_directories.load_images(target_samples_per_pixel, self.target_render_passes_usage)
      
      # TODO: Find a better way to deal with it! (DeepBlender)
      assert render_directories.have_loaded_images_identical_sizes()
      
      
      height, width = render_directories.size_of_loaded_images()
      
      # Split the images into tiles.
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
              features[Naming.source_feature_name(source_render_pass, index=source_index)] = TFRecordsCreator._bytes_feature(
                  tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
            source_index = source_index + 1
      
          # Prepare the target image tiles.
          target_render_directory = render_directories.samples_per_pixel_to_render_directories[target_samples_per_pixel][0]
          for target_render_pass in target_render_directory.render_pass_to_image:
            image = target_render_directory.render_pass_to_image[target_render_pass]
            features[Naming.target_feature_name(target_render_pass)] = TFRecordsCreator._bytes_feature(
                tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
          
          tfrecords_writer.write(features)
      
      render_directories.unload_images()
    tfrecords_writer.close()
  
  def create_statistics(self):
    tfrecords_statistics = TFRecordsStatistics(self)
    tfrecords_statistics.compute_and_save_statistics()
  
  @staticmethod
  def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  @staticmethod
  def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

  @staticmethod
  def _float_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class TFRecordsWriter:
  def __init__(self, name, base_directory, examples_per_tfrecords):
    self.name = name
    self.base_directory = base_directory
    self.examples_per_tfrecords = examples_per_tfrecords
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
    
    if self.added_tiles >= self.examples_per_tfrecords:
      self.close()
  
  def close(self):
    if self.writer != None:
      self.writer.close()
      TFRecordsWriter._compress(self.tfrecords_filename)
    self.added_tiles = 0
    self.tfrecords_filename = None
    self.writer = None
    self.tfrecords_index = self.tfrecords_index + 1
  
  @staticmethod
  def _compress(filename, delete_uncompressed=True):
    gzip_filename = filename + '.gz'
    original_file = open(filename, 'rb')
    gzip_file = gzip.open(gzip_filename, 'wb')
    gzip_file.writelines(original_file)
    gzip_file.close()
    original_file.close()
    if delete_uncompressed:
      os.remove(filename)

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
        mode_settings['tiles_height_width'], mode_settings['examples_per_tfrecords'])
    tfrecords_creators.append(tfrecords_creator)
  
  if not parsed_arguments.statistics:
    for tfrecords_creator in tfrecords_creators:
      tfrecords_creator.create_tfrecords()
  
  tf.enable_eager_execution()
  for tfrecords_creator in tfrecords_creators:
    tfrecords_creator.create_statistics()
  
if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)