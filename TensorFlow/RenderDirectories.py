from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from RenderDirectory import RenderDirectory

class RenderDirectories:
  def __init__(self, base_directory):
    self.base_directory = base_directory
    self.samples_per_pixel_to_render_directories = {}
    subdirectories = self._subdirectories(self.base_directory)
    for subdirectory in subdirectories:
      render_directory = RenderDirectory(subdirectory)
      samples_per_pixel = render_directory.samples_per_pixel
      if not samples_per_pixel in self.samples_per_pixel_to_render_directories:
        render_directories = [render_directory]
        self.samples_per_pixel_to_render_directories[samples_per_pixel] = render_directories
      else:
        render_directories = self.samples_per_pixel_to_render_directories[samples_per_pixel]
        render_directories.append(render_directory)

  def required_files_exist(self, samples_per_pixel, render_passes_usage):
    result = True
    if samples_per_pixel in self.samples_per_pixel_to_render_directories:
      render_directories = self.samples_per_pixel_to_render_directories[samples_per_pixel]
      for render_directory in render_directories:
        result = render_directory.required_files_exist(render_passes_usage)
        if not result:
          break
    return result

  def load_images(self, samples_per_pixel, render_passes_usage):
    if samples_per_pixel in self.samples_per_pixel_to_render_directories:
      render_directories = self.samples_per_pixel_to_render_directories[samples_per_pixel]
      for render_directory in render_directories:
        render_directory.load_images(render_passes_usage)
        
        # TODO: Horrible way to do it! (DeepBlender)
        assert render_directory.have_loaded_images_identical_sizes()
  
  def ground_truth_samples_per_pixel(self):
    result = 0
    for samples_per_pixel in self.samples_per_pixel_to_render_directories:
      if result < samples_per_pixel:
        result = samples_per_pixel
    return result
  
  def _subdirectories(self, directory):
    return filter(os.path.isdir, [os.path.join(directory, subdirectory) for subdirectory in os.listdir(directory)])
