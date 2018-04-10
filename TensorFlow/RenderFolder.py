from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

class RenderFolder:
  def __init__(self, folder_path, render_passes_usage):
    self.folder_path = folder_path
    self.render_passes_usage = render_passes_usage
    
    self.render_pass_to_image = {}
  
  def required_files_exist(self):
    # TODO: Using missing_render_pass_files in this way is not the best idea. (DeepBlender)
    self.missing_render_pass_files = []
    required_render_passes = self.render_passes_usage.render_passes()
    exr_files = self._exr_files(self.folder_path)
    for render_pass in required_render_passes:
      render_pass_exists = False
      for exr_file in exr_files:
        if render_pass in exr_file:
          render_pass_exists = True
          break
      if not render_pass_exists:
        self.missing_render_pass_files.append(render_pass)
    return len(self.missing_render_pass_files) == 0

  def load_images(self):
    self.render_pass_to_image = {}
    render_passes = self.render_passes_usage.render_passes()
    exr_files = self._exr_files(self.folder_path)
    for render_pass in render_passes:
      exr_loaded = False
      for exr_file in exr_files:
        if render_pass in exr_file:
          image = self._load_exr(exr_file)
          self.render_pass_to_image[render_pass] = image
          exr_loaded = True
          break
      if not exr_loaded:
        # TODO: Improve (DeepBlender)
        raise Exception('Image for \'' + render_pass + '\' could not be loaded or does not exist.')

  def _exr_files(self, folder_path):
    result = []
    for filename in os.listdir(folder_path):
      if filename.endswith('.exr'):
        result.append(os.path.join(folder_path, filename))
    return result

  def _load_exr(self, exr_path):
    try:
      image_type = cv2.IMREAD_UNCHANGED
      image = cv2.imread(exr_path, image_type)
      
      # REMARK: This dummy call avoids an error message (Assertion Failed)
      shape = image.shape
      
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if image.dtype != 'float32':
        image = image.astype(np.float32)
    except Exception:
      # TODO: Proper error handling (DeepBlender)
      print(exr_path)
    return image