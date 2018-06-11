from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Conv2dUtilities

class KernelPrediction:

  @staticmethod
  def kernel_prediction(inputs, kernel_inputs, kernel_size, use_softmax=True, mode='symmetric', data_format='channels_last'):

    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.height_width(inputs, data_format) == Conv2dUtilities.height_width(kernel_inputs, data_format)
    assert Conv2dUtilities.number_of_channels(kernel_inputs, data_format) == kernel_size ** 2
  
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    height_axis, width_axis = Conv2dUtilities.height_width_axis(inputs, data_format)
    number_of_channels = Conv2dUtilities.number_of_channels(inputs, data_format)
    pad = (kernel_size - 1) // 2
    
    if use_softmax:
      kernel_inputs = tf.nn.softmax(kernel_inputs, axis=channel_axis)
    inputs_split = tf.split(inputs, number_of_channels, axis=channel_axis)

    # TODO: Check whether there is a more efficient way than iterating through the channels one by one (DeepBlender)
    for index in range(number_of_channels):
      input = inputs_split[index]
      padding = []
      
      padded_input = Conv2dUtilities.pad_equally(input, pad, mode=mode, data_format=data_format)
      input_stack = []
      for i in range(kernel_size):
        for j in range(kernel_size):
          if Conv2dUtilities.is_batched(inputs):
            if data_format == 'channels_last':
              input_stack.append(padded_input[:, i:padded_input.shape[height_axis] - 2 * pad + i, j:padded_input.shape[width_axis] - 2 * pad + j, :])
            else:
              input_stack.append(padded_input[:, :, i:padded_input.shape[height_axis] - 2 * pad + i, j:padded_input.shape[width_axis] - 2 * pad + j])
          else:
            if data_format == 'channels_last':
              input_stack.append(padded_input[i:padded_input.shape[height_axis] - 2 * pad + i, j:padded_input.shape[width_axis] - 2 * pad + j, :])
            else:
              input_stack.append(padded_input[:, i:padded_input.shape[height_axis] - 2 * pad + i, j:padded_input.shape[width_axis] - 2 * pad + j])
      
      input_stack = tf.concat(input_stack, axis=channel_axis)
      input = tf.reduce_sum(tf.multiply(input_stack, kernel_inputs), axis=channel_axis, keepdims=True)
      
      inputs_split[index] = input
    inputs = tf.concat(inputs_split, axis=channel_axis)
    
    return inputs
