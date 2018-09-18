from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from OpenEXRDirectory import OpenEXRDirectory

class OpenEXRDirectories:

  def __init__(self, base_directory, number_of_sources_per_example, logger=None):
    self.base_directory = base_directory
    self.number_of_sources_per_example = number_of_sources_per_example
    self.samples_per_pixel_to_exr_directories = {}
    self.is_valid = True
    self.logger = logger

    if os.path.exists(self.base_directory):
      subdirectories = OpenEXRDirectories._subdirectories(self.base_directory)
      for subdirectory in subdirectories:
        exr_directory = OpenEXRDirectory(subdirectory, logger=self.logger)
        samples_per_pixel = exr_directory.samples_per_pixel
        if not samples_per_pixel in self.samples_per_pixel_to_exr_directories:
          exr_directories = [exr_directory]
          self.samples_per_pixel_to_exr_directories[samples_per_pixel] = exr_directories
        else:
          exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
          exr_directories.append(exr_directory)
          exr_directories.sort()
    else:
      self.is_valid = False
      if logger != None:
        logger.error('Base directory does not exist: ' + self.base_directory)

  def ensure_required_files_exist(self, number_of_sources_per_example, samples_per_pixel, render_passes_usage):
    if samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      if number_of_sources_per_example <= len(self.samples_per_pixel_to_exr_directories[samples_per_pixel]):
        exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
        for index, exr_directory in enumerate(exr_directories):
          if index < self.number_of_sources_per_example:
            exr_directory.ensure_required_files_exist(render_passes_usage)
            if not exr_directory.is_valid:
              self.is_valid = False
              break
      else:
        self.is_valid = False
        if self.logger != None:
          self.logger.error(
              self.base_directory + ' requires ' +
              str(number_of_sources_per_example) + ' subdirectories for ' +
              str(samples_per_pixel) + ' samples per pixel, but there is/are only ' +
              str(len(self.samples_per_pixel_to_exr_directories[samples_per_pixel])) + '.')
    else:
      self.is_valid = False
      if self.logger != None:
        self.logger.error(self.base_directory + ' does not have a subdirectory for ' + str(samples_per_pixel) + ' samples per pixel.')

  def load_images(self, samples_per_pixel, render_passes_usage):
    if samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
      for index, exr_directory in enumerate(exr_directories):
        if index < self.number_of_sources_per_example:
          exr_directory.load_images(render_passes_usage)
          if not exr_directory.is_valid:
            self.is_valid = False
            break
  
  def size_of_loaded_images(self):
    height = 0
    width = 0
    done = False
    for samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
      for exr_directory in exr_directories:
        if exr_directory.is_loaded():
          height, width = exr_directory.size_of_loaded_images()
          done = True
          break
      if done:
        break
    return height, width
  
  def ensure_loaded_images_identical_sizes(self):
    height, width = self.size_of_loaded_images()
    for samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
      for index, exr_directory in enumerate(exr_directories):
        if index < self.number_of_sources_per_example:
          exr_directory.ensure_loaded_images_have_size(height, width)
          if not exr_directory.is_valid:
            self.is_valid = False
            break
        else:
          break
      if not self.is_valid:
        break
  
  def unload_images(self):
    for samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      exr_directories = self.samples_per_pixel_to_exr_directories[samples_per_pixel]
      for exr_directory in exr_directories:
        exr_directory.unload_images()
  
  def ground_truth_samples_per_pixel(self):
    result = 0
    for samples_per_pixel in self.samples_per_pixel_to_exr_directories:
      if result < samples_per_pixel:
        result = samples_per_pixel
    return result
  
  @staticmethod
  def _subdirectories(directory):
    return filter(os.path.isdir, [os.path.join(directory, subdirectory) for subdirectory in os.listdir(directory)])
