from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

import json
import gzip

from RenderPasses import RenderPasses
from RenderPasses import RenderPassesUsage
from RenderDirectories import RenderDirectories


parser = argparse.ArgumentParser(description='Create tfrecords files for the DeepDenoiser.')
parser.add_argument('json_filename', help='The json specifying all the relevant details.')


class TFRecordsCreator:

  # TODO: Add log file functionality from previous version. (DeepBlender)
  # TODO: Add statistics from previous version. (DeepBlender)

  def __init__(
      self, name, base_tfrecords_directory, base_render_directory, relative_render_directories,
      source_samples_per_pixel, source_render_passes_usage,
      target_samples_per_pixel, target_render_passes_usage,
      tiles_height_width, tiles_per_tfrecords):
    self.name = name
    self.base_tfrecords_directory = base_tfrecords_directory
    self.source_samples_per_pixel = source_samples_per_pixel
    self.source_render_passes_usage = source_render_passes_usage
    self.target_samples_per_pixel = target_samples_per_pixel
    self.target_render_passes_usage = target_render_passes_usage
    self.tiles_height_width = tiles_height_width
    self.tiles_per_tfrecords = tiles_per_tfrecords
    
    self.render_directories_list = []
    for render_directories in relative_render_directories:
      new_render_directories = RenderDirectories(os.path.join(base_render_directory, render_directories))
      
      # TODO: Certainly not the best way to perform the validity checks. (DeepBlender)
      assert new_render_directories.required_files_exist(self.source_samples_per_pixel, self.source_render_passes_usage)
      
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel is 'best':
        target_samples_per_pixel = new_render_directories.ground_truth_samples_per_pixel()
      assert new_render_directories.required_files_exist(target_samples_per_pixel, self.target_render_passes_usage)
      
      self.render_directories_list.append(new_render_directories)
  
  def create(self):
    tfrecords_directory = os.path.join(self.base_tfrecords_directory, self.name)
    if not os.path.exists(tfrecords_directory):
      os.makedirs(tfrecords_directory)
    
    for render_directories in self.render_directories_list:
      target_samples_per_pixel = self.target_samples_per_pixel
      if target_samples_per_pixel is 'best':
        target_samples_per_pixel = render_directories.ground_truth_samples_per_pixel()
      
      render_directories.load_images(self.source_samples_per_pixel, self.source_render_passes_usage)
      render_directories.load_images(target_samples_per_pixel, self.target_render_passes_usage)
      
      # TODO: Find a better way to deal with it! (DeepBlender)
      assert render_directories.have_loaded_images_identical_sizes()
      
      # TODO: Processing
      
      render_directories.unload_images()
  
  def _compress(self, filename, delete_uncompressed=True):
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
        source_samples_per_pixel, source_render_passes_usage,
        target_samples_per_pixel, target_render_passes_usage,
        mode_settings['tiles_height_width'], mode_settings['tiles_per_tfrecords'])
    tfrecords_creators.append(tfrecords_creator)
    
  for tfrecords_creator in tfrecords_creators:
    tfrecords_creator.create()
  
  
if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)