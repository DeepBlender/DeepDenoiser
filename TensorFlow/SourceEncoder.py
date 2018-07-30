from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Conv2dUtilities import Conv2dUtilities

class SourceEncoder:

  # TODO: In depth testing! (DeepBlender)
  # The overall idea is to have multiple source inputs which were computed using different seeds.
  # Those are fed one after the other into the source encoder to refine and enhance the encoded source
  # which is then passed along to the actual denoiser.
  # Instead of combining the noisy sources into an encoded source, we may consider to have two input modules.
  # One for the prediction of a noisy source and the other one which also takes a noisy source, but an already
  # computed denoised one as well.

  @staticmethod
  def source_encoding(inputs, inputs_index, number_of_output_channels, encoded_source=None, activation_function=tf.nn.relu, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    # Logical implication 'a => b' has to be written as 'not (a) or (b)'
    assert not (encoded_source != None) or (Conv2dUtilities.has_valid_shape(encoded_source))
    assert not (encoded_source != None) or (
        Conv2dUtilities.height_width(encoded_source, data_format) == Conv2dUtilities.height_width(inputs, data_format))
    assert not (encoded_source != None) or (
        Conv2dUtilities.number_of_channels(encoded_source, data_format) == number_of_output_channels)
    
    height, width = Conv2dUtilities.height_width(inputs, data_format)
    
    if data_format == 'channels_last':
      inputs_weight_shape = [height, width, 1]
      encoded_source_shape = [height, width, number_of_output_channels]
    else:
      inputs_weight_shape = [1, height, width]
      encoded_source_shape = [number_of_output_channels, height, width]
    
    inputs_weight = tf.multiply(1. / (inputs_index + 1.), tf.ones(inputs_weight_shape, tf.float32))
    
    # Inputs need to consist of the original inputs, the inputs weight and the encoded source.
    
    if inputs_index == 0:
      encoded_source = tf.zeros(encoded_source_shape, tf.float32)
      
      # Adding the inputs weight and the encoded source does only work when the tensor is unbatched. This can be achieved with 'map_fn'.
      def add_weight_and_encoded_source(inputs):
        concat_axis = Conv2dUtilities.channel_axis(inputs, data_format)
        inputs = tf.concat([inputs, inputs_weight, encoded_source], concat_axis)
        return inputs

      inputs = tf.map_fn(add_weight_and_encoded_source, inputs)
    
    else:
      # Adding the inputs weight and the encoded source does only work when the tensor is unbatched. This can be achieved with 'map_fn'.
      def add_weight(inputs):
        concat_axis = Conv2dUtilities.channel_axis(inputs, data_format)
        inputs = tf.concat([inputs, inputs_weight], concat_axis)
        return inputs

      inputs = tf.map_fn(add_weight, inputs)
      
      concat_axis = Conv2dUtilities.channel_axis(inputs, data_format)
      inputs = tf.concat([inputs, encoded_source], concat_axis)
    
    
    number_of_convolutions = 2
    residual = inputs
    for _ in range(number_of_convolutions):
      residual = activation_function(residual)
      residual = tf.layers.conv2d(
        inputs=residual, filters=number_of_output_channels, kernel_size=(3, 3), padding='same',
        activation=None, data_format=data_format)
    
    result = tf.add(encoded_source, residual)
    return result
