import tensorflow as tf

class Conv2dUtilities:

  @staticmethod
  def has_valid_shape(inputs):
    return len(inputs.shape) == 3 or len(inputs.shape) == 4

  @staticmethod
  def is_batched(inputs):
    assert Conv2dUtilities.has_valid_shape(inputs)
    return len(inputs.shape) == 4

  @staticmethod
  def channel_axis(inputs, data_format):
    assert Conv2dUtilities.has_valid_shape(inputs)
    if data_format == 'channels_first':
      result = 0
    else:
      result = 2
    
    if Conv2dUtilities.is_batched(inputs):
      result = result + 1
    return result

  @staticmethod
  def number_of_channels(inputs, data_format):
    return inputs.shape[Conv2dUtilities.channel_axis(inputs, data_format)]

  @staticmethod
  def height_width_axis(inputs, data_format):
    assert Conv2dUtilities.has_valid_shape(inputs)
    if data_format == 'channels_first':
      result = 1
    else:
      result = 0
    
    if Conv2dUtilities.is_batched(inputs):
      result = result + 1
    return result, result + 1

  @staticmethod
  def height_width(inputs, data_format):
    height_index, width_index = Conv2dUtilities.height_width_axis(inputs, data_format)
    height = inputs.shape[height_index]
    width = inputs.shape[width_index]
    return height, width

  @staticmethod
  def convert_to_data_format(inputs, data_format):
    assert Conv2dUtilities.has_valid_shape(inputs)
    if Conv2dUtilities.is_batched(inputs):
      if data_format == 'channels_first':
        # channels_last to channels_first
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
      else:
        # channels_first to channels_last
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
      if data_format == 'channels_first':
        # channels_last to channels_first
        inputs = tf.transpose(inputs, [2, 0, 1])
      else:
        # channels_first to channels_last
        inputs = tf.transpose(inputs, [1, 2, 0])
    return inputs
  
  @staticmethod
  def non_zero_mask(inputs, data_format):
    channel_index = Conv2dUtilities.channel_axis(inputs, data_format)
    inputs = tf.abs(inputs)
    inputs = tf.reduce_sum(inputs, axis=channel_index)
    inputs = tf.sign(inputs)
    return inputs

  @staticmethod
  def pad_equally(inputs, pad, mode='symmetric', data_format='channels_last'):
    '''
    Pad equally on the left, right, top and bottom.
    '''
    
    paddings = []
    channel_index = Conv2dUtilities.channel_axis(inputs, data_format)
    if Conv2dUtilities.is_batched(inputs):
      paddings.append([0, 0])
      channel_index = channel_index - 1
    
    for i in range(3):
      if i == channel_index:
        paddings.append([0, 0])
      else:
        paddings.append([pad, pad])
    
    result = tf.pad(inputs, paddings, mode)
    return result
