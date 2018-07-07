import tensorflow as tf

from Conv2dUtilities import Conv2dUtilities
from RenderPasses import RenderPasses

class DataAugmentation:
  
  @staticmethod
  def flip_left_right(inputs, name, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = DataAugmentation._convert_to_channels_last(inputs)
    
    # Flip
    inputs = tf.image.flip_left_right(inputs)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = DataAugmentation._flip_screen_space_normals(inputs)
    if name == RenderPasses.NORMAL:
      inputs = DataAugmentation._flip_normals(inputs)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = DataAugmentation._convert_to_channels_first(inputs)
    
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
  def rotate90(inputs, k, name, data_format='channels_last'):
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = DataAugmentation._convert_to_channels_last(inputs)
    
    # Rotate
    inputs = tf.image.rot90(inputs, k=k)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = DataAugmentation._rotate90_screen_space_normals(inputs, k)
    if name == RenderPasses.NORMAL:
      inputs = DataAugmentation._rotate90_normals(inputs, k)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = DataAugmentation._convert_to_channels_first(inputs)
    
    return inputs
    
  @staticmethod
  def _rotate90_screen_space_normals(inputs, k):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(inputs, [1, 1, 1], channel_axis)
    if k == 1:
      # x -> -y
      # y -> x
      temporary_screen_space_normal_x = screen_space_normal_x
      screen_space_normal_x = tf.negative(screen_space_normal_y)
      screen_space_normal_y = temporary_screen_space_normal_x
    elif k == 2:
      # x -> -x
      # y -> -y
      screen_space_normal_x = tf.negative(screen_space_normal_x)
      screen_space_normal_y = tf.negative(screen_space_normal_y)
    elif k == 3:
      # x -> y
      # y -> -x
      temporary_screen_space_normal_y = screen_space_normal_y
      screen_space_normal_y = tf.negative(screen_space_normal_x)
      screen_space_normal_x = temporary_screen_space_normal_y
    
    inputs = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], channel_axis)
    return(inputs)
  
  @staticmethod
  def _rotate90_normals(inputs, k):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    raise Exception('TODO: Rotate90 normals is not yet implemented. (DeepBlender)')
    
    return(inputs)
  
  @staticmethod
  def rotate90_normals(inputs, k, data_format):
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    
    raise Exception('TODO: Rotate90 normals is not yet implemented. (DeepBlender)')
    
    return(inputs)
  
  @staticmethod
  def permute_rgb(inputs, permutation, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    #assert len(permutation) == 3
    #assert (0 in permutation) and (1 in permutation) and (2 in permutation)
    
    # Swap colors.
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    
    colors = tf.split(inputs, [1, 1, 1], channel_axis)
    inputs = tf.concat([colors[permutation[0]], colors[permutation[1]], colors[permutation[2]]], channel_axis)
    
    return inputs
  
  @staticmethod
  def _convert_to_channels_last(inputs):
    if Conv2dUtilities.is_batched(inputs):
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    else:
      inputs = tf.transpose(inputs, [2, 0, 1])
    return inputs
  
  @staticmethod
  def _convert_to_channels_first(inputs):
    if Conv2dUtilities.is_batched(inputs):
      inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
      inputs = tf.transpose(inputs, [1, 2, 0])
    return inputs
