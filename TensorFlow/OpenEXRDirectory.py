from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from RenderPasses import RenderPasses

class OpenEXRDirectory:

  def __init__(self, directory, logger=None):
    self.directory = directory
    self.unload_images()
    self.logger = logger
    self.is_valid = True
    
    # This is ensured by the Blender script.
    self.samples_per_pixel = int(self.directory.split('_')[-3])
  
  def __lt__(self, other):
    return self.directory < other.directory

  def _gt__(self, other):
    return self.directory > other.directory

  def ensure_required_files_exist(self, render_passes_usage):
    required_render_passes = render_passes_usage.render_passes()
    exr_files = OpenEXRDirectory._exr_files(self.directory)
    for render_pass in required_render_passes:
      render_pass_exists = False
      is_render_pass_file_unique = True
      for exr_file in exr_files:
        # HACK: We add the _ to distinguish between the normal and screen space normal pass.
        if '_' + render_pass + '_' in exr_file:
          if render_pass_exists:
            is_render_pass_file_unique = False
            break
          else:
            render_pass_exists = True
      if not render_pass_exists:
        self.is_valid = False
        if self.logger != None:
          self.logger.error(self.directory + ' does not contain an exr file for ' + render_pass + '.')
        break
      if not is_render_pass_file_unique:
        self.is_valid = False
        if self.logger != None:
          self.logger.error(
              'There is more than one file in ' + self.directory + ' which could be used for the ' +
              render_pass + ' pass.')

  def load_images(self, render_passes_usage):
    self.render_passes_usage = render_passes_usage
    self.render_pass_to_image = {}
    render_passes = self.render_passes_usage.render_passes()
    exr_files = OpenEXRDirectory._exr_files(self.directory)
    for render_pass in render_passes:
      exr_loaded = False
      for exr_file in exr_files:
        # HACK: We add the _ to distinguish between the normal and screen space normal pass.
        if '_' + render_pass + '_' in exr_file:
          image = OpenEXRDirectory._load_exr(exr_file)
          
          # Special cases: Alpha and depth passes only have one channel.
          if RenderPasses.number_of_channels(render_pass) == 1:
            image = image[:, :, 0]
          
          self.render_pass_to_image[render_pass] = image

          # Neither NaN, nor infinity is valid.
          if not np.isfinite(image).all():
            self.is_valid = False
            if self.logger != None:
              self.logger.error('There is at least one value in ' + exr_file + ' which is not finite.')

          exr_loaded = True
          break
      if not self.is_valid:
        break
      
      if not exr_loaded:
        # This should never happen, because we ensure_required_files_exist. (DeepBlender)
        raise Exception('Image for \'' + render_pass + '\' could not be loaded or does not exist.')

  def is_loaded(self):
    return self.render_passes_usage != None
  
  def size_of_loaded_images(self):
    height = 0
    width = 0
    for render_pass in self.render_pass_to_image:
      image = self.render_pass_to_image[render_pass]
      height = image.shape[0]
      width = image.shape[1]
      break
    return height, width

  def ensure_loaded_images_have_size(self, height, width):
    for render_pass in self.render_pass_to_image:
      image = self.render_pass_to_image[render_pass]
      image_height = image.shape[0]
      image_width = image.shape[1]
      if image_height != height or image_width != width:
        self.is_valid = False
        if self.logger != None:
          self.logger.error(
              render_pass + ' from ' + self.directory + ' does not have an expected size of (' + 
              str(width) + ', ' + str(height) + '), but (' +
              str(image_width) + ', ' + str(image_height) + ').')
  
  def unload_images(self):
    self.render_passes_usage = None
    self.render_pass_to_image = {}

  @staticmethod
  def _exr_files(directory):
    result = []
    for filename in os.listdir(directory):
      if filename.endswith('.exr'):
        result.append(os.path.join(directory, filename))
    return result

  @staticmethod
  def _load_exr(exr_path):
    try:

      # image_type = cv2.IMREAD_UNCHANGED
      # image = cv2.imread(exr_path, image_type)


      # Images have to be loaded indirectly to allow utf-8 paths.

      stream = open(exr_path, "rb")
      bytes = bytearray(stream.read())
      np_array = np.asarray(bytes, dtype=np.uint8)
      
      image_type = cv2.IMREAD_UNCHANGED
      image = cv2.imdecode(np_array, image_type)

      
      # REMARK: This dummy call avoids an error message (Assertion Failed)
      shape = image.shape
      
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if image.dtype != 'float32':
        image = image.astype(np.float32)
    except Exception:
      # TODO: Proper error handling (DeepBlender)
      print(exr_path)
    return image