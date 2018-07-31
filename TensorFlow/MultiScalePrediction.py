from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Conv2dUtilities import Conv2dUtilities

class MultiScalePrediction:

  @staticmethod
  def scale_down(inputs, heigh_width_scale_factor=2, data_format='channels_last'):
    result = tf.layers.average_pooling2d(inputs, heigh_width_scale_factor, heigh_width_scale_factor, padding='same', data_format=data_format)
    return result
  
  @staticmethod
  def scale_up(inputs, height_width_scale_factor=2, data_format='channels_last'):
  
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
    
    height, width = Conv2dUtilities.height_width(inputs, 'channels_last')
    height = height * height_width_scale_factor
    width = width * height_width_scale_factor
    
    # TODO: Check whether we need to align the corners (DeepBlender)
    inputs = tf.image.resize_images(inputs, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs
  
  @staticmethod
  def compose_scales(small_inputs, inputs, activation_function=tf.nn.relu, data_format='channels_last'):
  
    # TODO: Find better names (DeepBlender)
    # Small -> Coarse
  
    small_inputs = MultiScalePrediction.scale_up(small_inputs, data_format=data_format)
    
    low_frequency_weights = MultiScalePrediction._compose_scales_neural_network(small_inputs, inputs, activation_function=activation_function, data_format=data_format)
    
    low_frequency_inputs = MultiScalePrediction.scale_down(inputs, data_format=data_format)
    low_frequency_inputs = MultiScalePrediction.scale_up(low_frequency_inputs, data_format=data_format)

    weighted_low_frequency_inputs = tf.multiply(low_frequency_weights, low_frequency_inputs)
    weighted_small_inputs = tf.multiply(low_frequency_weights, small_inputs)
    
    # Subtract the weighted low frequency from the inputs and add the weighted one from the small inputs instead.
    inputs = tf.add(tf.subtract(inputs, weighted_low_frequency_inputs), weighted_small_inputs)
    
    return inputs
  
  @staticmethod
  def _compose_scales_neural_network(small_inputs, inputs, activation_function, data_format='channels_last'):
    concat_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    inputs = tf.concat([small_inputs, inputs], concat_axis)
    
    # TODO: Make this number configurable? (DeepBlender)
    number_of_filters = 24
    
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=number_of_filters, kernel_size=(1, 1), padding='same',
        activation=activation_function, data_format=data_format)
    
    # TODO: Make this number configurable? (DeepBlender)
    number_of_residual_blocks = 2
    for _ in range(number_of_residual_blocks):
      inputs = MultiScalePrediction._residual_block(inputs, activation_function, data_format=data_format)
    
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=1, kernel_size=(1, 1), padding='same',
        activation=activation_function, data_format=data_format)
    
    inputs = tf.sigmoid(inputs)
    return inputs
  
  @staticmethod
  def _residual_block(inputs, activation_function, data_format='channels_last'):
    residual = inputs
    
    number_of_filters = Conv2dUtilities.number_of_channels(inputs, data_format)
    number_of_convolutions = 2
    for _ in range(number_of_convolutions):
      residual = activation_function(residual)
      residual = tf.layers.conv2d(
        inputs=residual, filters=number_of_filters, kernel_size=(3, 3), padding='same',
        activation=None, data_format=data_format)
      
    inputs = tf.add(inputs, residual)
    return inputs
