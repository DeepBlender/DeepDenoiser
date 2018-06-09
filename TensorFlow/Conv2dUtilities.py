import tensorflow as tf

def has_valid_shape(inputs):
  return len(inputs.shape) == 3 or len(inputs.shape) == 4

def is_batched(inputs):
  assert has_valid_shape(inputs)
  return len(inputs.shape) == 4

def channel_axis(inputs, data_format):
  assert has_valid_shape(inputs)
  if data_format == 'channels_first':
    result = 0
  else:
    result = 2
  
  if is_batched(inputs):
    result = result + 1
  return result

def number_of_channels(inputs, data_format):
  return inputs.shape[channel_axis(inputs, data_format)]

def height_width_axis(inputs, data_format):
  assert has_valid_shape(inputs)
  if data_format == 'channels_first':
    result = 1
  else:
    result = 0
  
  if is_batched(inputs):
    result = result + 1
  return result, result + 1

def height_width(inputs, data_format):
  height_index, width_index = height_width_axis(inputs, data_format)
  height = inputs.shape[height_index]
  width = inputs.shape[width_index]
  return height, width

def non_zero_mask(inputs, data_format):
  channel_index = channel_axis(inputs, data_format)
  inputs = tf.abs(inputs)
  inputs = tf.reduce_sum(inputs, axis=channel_index)
  inputs = tf.sign(inputs)
  return inputs

def pad_equally(inputs, pad, mode='symmetric', data_format='channels_last'):
  '''
  Pad equally on the left, right, top and bottom.
  '''
  
  paddings = []
  channel_index = channel_axis(inputs, data_format)
  if is_batched(inputs):
    paddings.append([0, 0])
    channel_index = channel_index - 1
  
  for i in range(3):
    if i == channel_index:
      paddings.append([0, 0])
    else:
      paddings.append([pad, pad])
  
  result = tf.pad(inputs, paddings, mode)
  return result
