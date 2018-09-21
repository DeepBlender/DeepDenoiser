from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging

import tensorflow as tf

import json
import gzip

from Naming import Naming
from RenderPasses import RenderPassesUsage
from TFRecordsStatistics import TFRecordsStatistics
from OpenEXRDirectories import OpenEXRDirectories


parser = argparse.ArgumentParser(description='Create tfrecords files for the DeepDenoiser.')

parser.add_argument(
    'json_filename',
    help='The json specifying all the relevant details.')

parser.add_argument(
    '--statistics', action="store_true",
    help='Only recalculate the statistics.')

class TFRecordsCreator:

  def __init__(
      self, name, base_tfrecords_directory, base_exr_directory, relative_exr_directories,
      source_samples_per_pixel_list, source_render_passes_usage, number_of_sources_per_example,
      target_samples_per_pixel, target_render_passes_usage,
      tiles_height_width, examples_per_tfrecords,
      group_by_samples_per_pixel):
    self.name = name
    self.base_tfrecords_directory = base_tfrecords_directory
    self.source_samples_per_pixel_list = source_samples_per_pixel_list
    self.source_render_passes_usage = source_render_passes_usage
    self.number_of_sources_per_example = number_of_sources_per_example
    self.target_samples_per_pixel = target_samples_per_pixel
    self.target_render_passes_usage = target_render_passes_usage
    self.tiles_height_width = tiles_height_width
    self.examples_per_tfrecords = examples_per_tfrecords
    self.group_by_samples_per_pixel = group_by_samples_per_pixel

    if not os.path.exists(self.base_tfrecords_directory):
      os.makedirs(self.base_tfrecords_directory)


    # Logger
    logger_filename = os.path.join(self.base_tfrecords_directory, self.name + '.log')

    self.logger = logging.getLogger(filename)
    self.logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s (%(asctime)s): %(message)s')
    file_handler = logging.FileHandler(logger_filename, mode='w+')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    self.logger.addHandler(file_handler)
    

    self.exr_directories_list = []
    for exr_directories in relative_exr_directories:
      new_exr_directories = OpenEXRDirectories(os.path.join(base_exr_directory, exr_directories), self.number_of_sources_per_example, self.logger)
      
      # Some validity checks.
      if new_exr_directories.is_valid:
        for source_samples_per_pixel in self.source_samples_per_pixel_list:
          new_exr_directories.ensure_required_files_exist(
              self.number_of_sources_per_example, source_samples_per_pixel, self.source_render_passes_usage)
          if not new_exr_directories.is_valid:
            break
        
        if new_exr_directories.is_valid:
          target_samples_per_pixel = self.target_samples_per_pixel
          if target_samples_per_pixel == 'best':
            target_samples_per_pixel = new_exr_directories.ground_truth_samples_per_pixel()
          new_exr_directories.ensure_required_files_exist(
              1, target_samples_per_pixel, self.target_render_passes_usage)
          if not new_exr_directories.is_valid:
            break
        
        if new_exr_directories.is_valid:
          self.exr_directories_list.append(new_exr_directories)
  
  def create_tfrecords(self):
    source_samples_per_pixel_lists = []
    if self.group_by_samples_per_pixel:
      for source_samples_per_pixel_list in self.source_samples_per_pixel_list:
        source_samples_per_pixel_lists.append([source_samples_per_pixel_list])
    else:
      source_samples_per_pixel_lists.append(self.source_samples_per_pixel_list)

    for source_samples_per_pixel_list in source_samples_per_pixel_lists:
      tfrecords_writer = TFRecordsWriter(
          self.name, self.base_tfrecords_directory, self.examples_per_tfrecords,
          self.group_by_samples_per_pixel, source_samples_per_pixel_list)

      for exr_directories in self.exr_directories_list:
        target_samples_per_pixel = self.target_samples_per_pixel
        if target_samples_per_pixel == 'best':
          target_samples_per_pixel = exr_directories.ground_truth_samples_per_pixel()
        
        for source_samples_per_pixel in source_samples_per_pixel_list:
          exr_directories.load_images(source_samples_per_pixel, self.source_render_passes_usage)
          if not exr_directories.is_valid:
            break
        if exr_directories.is_valid:
          exr_directories.load_images(target_samples_per_pixel, self.target_render_passes_usage)
        
        # Simple validity checks.
        if exr_directories.is_valid:
          exr_directories.ensure_loaded_images_identical_sizes()
        
        if exr_directories.is_valid:

          # TODO: Maybe which image parts are contained in which tfrecords. (DeepBlender)

          height, width = exr_directories.size_of_loaded_images()
          
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
              for source_samples_per_pixel in source_samples_per_pixel_list:
                for index, source_exr_directory in enumerate(
                    exr_directories.samples_per_pixel_to_exr_directories[source_samples_per_pixel]):
                  if index < self.number_of_sources_per_example:
                    for source_render_pass in source_exr_directory.render_pass_to_image:
                      source_feature_name = Naming.source_feature_name(
                          source_render_pass,
                          samples_per_pixel=source_samples_per_pixel,
                          index=index)
                      image = source_exr_directory.render_pass_to_image[source_render_pass]
                      features[source_feature_name] = TFRecordsCreator._bytes_feature(
                              tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
          
              # Prepare the target image tiles.
              target_exr_directory = exr_directories.samples_per_pixel_to_exr_directories[target_samples_per_pixel][0]
              for target_render_pass in target_exr_directory.render_pass_to_image:
                image = target_exr_directory.render_pass_to_image[target_render_pass]
                features[Naming.target_feature_name(target_render_pass)] = TFRecordsCreator._bytes_feature(
                    tf.compat.as_bytes(image[x1:x2, y1:y2].tostring()))
              
              tfrecords_writer.write(features)
        
        exr_directories.unload_images()
      tfrecords_writer.close()


      # Save the settings.
      
      settings = {}
      settings['tiles_height_width'] = self.tiles_height_width
      settings['number_of_sources_per_example'] = self.number_of_sources_per_example
      settings['source_samples_per_pixel_list'] = source_samples_per_pixel_list

      filename = self.name + '.json'
      if self.group_by_samples_per_pixel:
        filename = self.name + '_' + str(source_samples_per_pixel_list[0]) + '.json'

      settings_json_filename = os.path.join(self.base_tfrecords_directory, filename)
      settings_json_content = json.dumps(settings, cls=DataSettingsEncoder, sort_keys=True, indent=2)
      with open(settings_json_filename, 'w+', encoding='utf-8') as settings_json_file:
        settings_json_file.write(settings_json_content)
  
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
  def __init__(
      self, name, base_directory, examples_per_tfrecords,
      group_by_samples_per_pixel, source_samples_per_pixel_list):
    self.name = name
    self.base_directory = base_directory
    self.examples_per_tfrecords = examples_per_tfrecords
    self.group_by_samples_per_pixel = group_by_samples_per_pixel
    self.source_samples_per_pixel_list = source_samples_per_pixel_list
    self.tfrecords_directory = os.path.join(self.base_directory, self.name)
    if self.group_by_samples_per_pixel:
      assert len(self.source_samples_per_pixel_list) == 1
      self.tfrecords_directory = os.path.join(
          self.tfrecords_directory, str(self.source_samples_per_pixel_list[0]))
    if not os.path.exists(self.tfrecords_directory):
      os.makedirs(self.tfrecords_directory)
    self.writer = None
    self.tfrecords_index = 0
    self.added_tiles = 0
  
  def write(self, features):
    if self.writer == None:
      self.tfrecords_filename = os.path.join(
          self.tfrecords_directory, self.name + '_' + str(self.tfrecords_index) + '.tfrecords')
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

class DataSettingsEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, '__json__'):
      return obj.__json__()
    if hasattr(obj, '__dict__'):
      return obj.__dict__
    return json.JSONEncoder.default(self, obj)

def main(parsed_arguments):
  try:
    json_filename = parsed_arguments.json_filename
    absolute_json_directory = os.path.dirname(os.path.abspath(json_filename))
    json_content = open(json_filename, 'r', encoding='utf-8').read()
    parsed_json = json.loads(json_content)
  except:
    print('Expected a valid json file as argument.')
  
  # Directories have to be relative to the json's path if they are not absolute.
  base_exr_directory = parsed_json['base_exr_directory']
  if not os.path.isabs(base_exr_directory):
    base_exr_directory = os.path.join(absolute_json_directory, base_exr_directory)
    base_exr_directory = os.path.realpath(base_exr_directory)

  base_tfrecords_directory = parsed_json['base_tfrecords_directory']
  if not os.path.isabs(base_tfrecords_directory):
    base_tfrecords_directory = os.path.join(absolute_json_directory, base_tfrecords_directory)
    base_tfrecords_directory = os.path.realpath(base_tfrecords_directory)

  if not os.path.exists(base_tfrecords_directory):
    os.makedirs(base_tfrecords_directory)

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

  # Central error logger
  logger_filename = os.path.join(base_tfrecords_directory, 'Error.log')
  logging.basicConfig(
      level=logging.ERROR,
      format='%(levelname)s, %(name)s (%(asctime)s): %(message)s',
      handlers=[
          logging.FileHandler(logger_filename, mode='w+'),
          logging.StreamHandler()])

  tfrecords_creators = []
  for mode_name in mode_name_to_mode_settings:
    mode_settings = mode_name_to_mode_settings[mode_name]
    tfrecords_creator = TFRecordsCreator(
        mode_name, base_tfrecords_directory, base_exr_directory, mode_settings['exr_directories'],
        source_samples_per_pixel, source_render_passes_usage, number_of_sources_per_example,
        target_samples_per_pixel, target_render_passes_usage,
        mode_settings['tiles_height_width'], mode_settings['examples_per_tfrecords'],
        mode_settings['group_by_samples_per_pixel'])
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