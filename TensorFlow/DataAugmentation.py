import tensorflow as tf

from Conv2dUtilities import Conv2dUtilities
from RenderPasses import RenderPasses

class DataAugmentation:
  
  @staticmethod
  def flip_left_right(inputs, name, flip, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
    
    # Flip
    inputs = tf.cond(flip > 0, lambda: tf.image.flip_left_right(inputs), lambda: inputs)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = tf.cond(flip > 0, lambda: DataAugmentation._flip_screen_space_normals(inputs), lambda: inputs)
    if name == RenderPasses.NORMAL:
      inputs = tf.cond (flip > 0, lambda: DataAugmentation._flip_normals(inputs), lambda: inputs)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs
  
  @staticmethod
  def _flip_screen_space_normals(inputs):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(inputs, [1, 1, 1], channel_axis)
    screen_space_normal_x = tf.negative(screen_space_normal_x)
    inputs = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], channel_axis)
    return(inputs)
  
  @staticmethod
  def _flip_normals(inputs):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    raise Exception('TODO: Normals flipping is not yet implemented. (DeepBlender)')
    
    return(inputs)
  
  @staticmethod
  def rotate_90(inputs, k, name, data_format='channels_last'):
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
    
    # Rotate
    inputs = tf.image.rot90(inputs, k=k)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = DataAugmentation._rotate_90_screen_space_normals(inputs, k)
    if name == RenderPasses.NORMAL:
      inputs = DataAugmentation._rotate_90_normals(inputs, k)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs
    
  @staticmethod
  def _rotate_90_screen_space_normals(inputs, k):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(inputs, [1, 1, 1], channel_axis)
    
    def _rotate_90(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> -y
      # y -> x
      temporary_screen_space_normal_x = screen_space_normal_x
      screen_space_normal_x = tf.negative(screen_space_normal_y)
      screen_space_normal_y = temporary_screen_space_normal_x
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    def _rotate_180(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> -x
      # y -> -y
      screen_space_normal_x = tf.negative(screen_space_normal_x)
      screen_space_normal_y = tf.negative(screen_space_normal_y)
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    def _rotate_270(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> y
      # y -> -x
      temporary_screen_space_normal_y = screen_space_normal_y
      screen_space_normal_y = tf.negative(screen_space_normal_x)
      screen_space_normal_x = temporary_screen_space_normal_y
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    cases =[
        (tf.equal(k, 1), lambda: _rotate_90(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z)),
        (tf.equal(k, 2), lambda: _rotate_180(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z)),
        (tf.equal(k, 3), lambda: _rotate_270(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z))]
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.case(
        cases, default=lambda: (screen_space_normal_x, screen_space_normal_y, screen_space_normal_z), exclusive=True)
    
    inputs = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], channel_axis)
    return(inputs)
  
  @staticmethod
  def _rotate_90_normals(inputs, k):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    raise Exception('TODO: Rotate 90 normals is not yet implemented. (DeepBlender)')
    
    return(inputs)
  
  @staticmethod
  def permute_rgb(inputs, permute, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    def _permute_rgb(inputs, permutation):
      channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
      result = tf.split(inputs, [1, 1, 1], channel_axis)
      result = tf.concat([result[permutation[0]], result[permutation[1]], result[permutation[2]]], channel_axis)
      return result
    
    cases =[
        (tf.equal(permute, 1), lambda: _permute_rgb(inputs, [0, 2, 1])),
        (tf.equal(permute, 2), lambda: _permute_rgb(inputs, [1, 0, 2])),
        (tf.equal(permute, 3), lambda: _permute_rgb(inputs, [1, 2, 0])),
        (tf.equal(permute, 4), lambda: _permute_rgb(inputs, [2, 0, 1])),
        (tf.equal(permute, 5), lambda: _permute_rgb(inputs, [2, 1, 0]))]
    inputs = tf.case(cases, default=lambda: inputs, exclusive=True)
    
    return inputs


class DataAugmentationUsage:

  def __init__(self, use_rotate_90, use_flip_left_right, use_rgb_permutation):
    self.use_rotate_90 = use_rotate_90
    self.use_flip_left_right = use_flip_left_right
    self.use_rgb_permutation = use_rgb_permutation
