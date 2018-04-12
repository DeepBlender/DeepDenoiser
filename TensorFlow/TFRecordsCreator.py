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


def mode_name_to_render_directories(parsed_json):
  result = {}

  base_render_directories = parsed_json['base_render_directories']
  mode_name_to_mode_settings = parsed_json['modes']
  
  source = parsed_json['source']
  source_samples_per_pixel = source['samples_per_pixel']
  source_render_passes_usage = RenderPassesUsage()
  source_render_passes_usage.__dict__ = source['features']
  
  target = parsed_json['target']
  target_samples_per_pixel = target['samples_per_pixel']
  target_render_passes_usage = RenderPassesUsage()
  target_render_passes_usage.__dict__ = target['features']
  
  for mode_name in mode_name_to_mode_settings:
    mode_settings = mode_name_to_mode_settings[mode_name]
    relative_render_directories = mode_settings['render_directories']
    render_directories_result = []
    for relative_directory in relative_render_directories:
      render_directories = RenderDirectories(os.path.join(base_render_directories, relative_directory))
      
      # TODO: Certainly not the best way to perform the validity checks. (DeepBlender)
      assert render_directories.required_files_exist(source_samples_per_pixel, source_render_passes_usage)
      
      samples_per_pixel = target_samples_per_pixel
      if samples_per_pixel is 'best':
        samples_per_pixel = render_directories.ground_truth_samples_per_pixel()
      assert render_directories.required_files_exist(samples_per_pixel, target_render_passes_usage)
      
      render_directories_result.append(render_directories)
    result[mode_name] = render_directories_result

  return result

def main(parsed_arguments):
  try:
    json_filename = parsed_arguments.json_filename
    json_content = open(json_filename, 'r').read()
    parsed_json = json.loads(json_content)
  except:
    print('Expected a valid json file as argument.')
  
  mode_names_to_render_directories = mode_name_to_render_directories(parsed_json)
  
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
  

if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)