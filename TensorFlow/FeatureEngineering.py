from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Conv2dUtilities import Conv2dUtilities

class FeatureEngineering:

  def _local_mean(inputs, variance_mode='uniform', data_format='channels_last'):
    kernel_size = 3
    padding_mode='symmetric'
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    height_axis, width_axis = Conv2dUtilities.height_width_axis(inputs, data_format)
    number_of_channels = Conv2dUtilities.number_of_channels(inputs, data_format)
    is_batched = Conv2dUtilities.is_batched(inputs)
    pad = (kernel_size - 1) // 2
    
    if data_format == 'channels_last':
      short_data_format = 'NHWC'
    else:
      short_data_format = 'NCHW'
    
    padded_inputs = Conv2dUtilities.pad_equally(inputs, pad, mode=padding_mode, data_format=data_format)
    padded_inputs_split = tf.split(padded_inputs, number_of_channels, axis=channel_axis)
    joined_inputs = []
    
    filter_shape = [kernel_size, kernel_size, 1, 1]
    if variance_mode == 'uniform':
      filter = tf.ones(filter_shape)
    else:
      assert variance_mode == 'neighbor'
      
      assert kernel_size == 3
      filter = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
      filter = tf.reshape(filter, [kernel_size, kernel_size, 1, 1])
      
    filter = tf.divide(filter, tf.reduce_sum(filter))
    
    for index in range(number_of_channels):
      padded_input = padded_inputs_split[index]
      if not is_batched:
        padded_input = tf.stack([padded_input])
      
      input = tf.nn.conv2d(padded_input, filter=filter, strides=[1, 1, 1, 1], padding='VALID', data_format=short_data_format)
      
      if not is_batched:
        input = input[0]
      joined_inputs.append(input)
    
    inputs = tf.concat(joined_inputs, axis=channel_axis)
    
    return inputs
  
  def variance(inputs, variance_mode='uniform', relative_variance=False, compress_to_one_channel=False, epsilon=1e-4, data_format='channels_last'):
    mean_of_inputs = FeatureEngineering._local_mean(inputs, variance_mode=variance_mode, data_format=data_format)
    squared_mean_of_inputs = tf.square(mean_of_inputs)
    mean_of_squared_inputs = FeatureEngineering._local_mean(tf.square(inputs), variance_mode=variance_mode, data_format=data_format)
    result = tf.subtract(mean_of_squared_inputs, squared_mean_of_inputs)
    
    if relative_variance:
      result = tf.divide(result, tf.maximum(squared_mean_of_inputs, epsilon))
    
    if compress_to_one_channel:
      channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
      result = tf.reduce_mean(result, axis=channel_axis, keepdims=True)
    
    return result
