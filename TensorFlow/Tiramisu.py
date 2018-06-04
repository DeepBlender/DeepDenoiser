#
# Based on:
#   The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
#   https://arxiv.org/abs/1611.09326
#
# Contrary to the original paper, this implementation does not include
# normalization or dropout.
#

import tensorflow as tf
import Conv2dUtilities

class Tiramisu:

  def __init__(
      self, number_of_preprocessing_convolution_filters, number_of_filters_for_convolution_blocks,
      number_of_convolutions_per_block, number_of_output_filters,
      activation_function=tf.nn.relu, data_format='channels_last'):

    self.number_of_preprocessing_convolution_filters = number_of_preprocessing_convolution_filters
    self.number_of_filters_for_convolution_blocks = number_of_filters_for_convolution_blocks
    self.number_of_convolutions_per_block = number_of_convolutions_per_block
    self.number_of_output_filters = number_of_output_filters
    self.activation_function = activation_function
    self.data_format = data_format

  def __convolution_block(self, inputs, number_of_filters, block_name):
    concat_axis = Conv2dUtilities.channel_axis(inputs, self.data_format)
    with tf.name_scope('convolution_block_' + block_name):
      for i in range(self.number_of_convolutions_per_block):
        with tf.name_scope('convolution_' + block_name + '_' + str(i + 1)):
          temporary = tf.layers.conv2d(
              inputs=inputs, filters=number_of_filters, kernel_size=(3, 3), padding='same',
              activation=self.activation_function, data_format=self.data_format)
          inputs = tf.concat([inputs, temporary], concat_axis)
    return inputs

  def __downsample(self, inputs):
    # TODO: Make the downsampling configurable (DeepBlender)
    number_of_filters = Conv2dUtilities.number_of_channels(inputs, self.data_format)
    with tf.name_scope('downsample'):
      
      # REMARK:
      # In the paper, they also use a 1x1 convolution before the actual downsampling.
      # We skip the batch normalization though.
      inputs = tf.layers.conv2d(
          inputs=inputs, filters=number_of_filters, kernel_size=(1, 1), padding='same',
          activation=self.activation_function, data_format=self.data_format)
      
      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=(2, 2), strides=(2, 2), padding='same',
          data_format=self.data_format)
    return inputs

  def __upsample(self, inputs, number_of_filters):
    with tf.name_scope('upsample'):
      inputs = tf.layers.conv2d_transpose(
          inputs=inputs, filters=number_of_filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
          activation=self.activation_function, data_format=self.data_format)
      return inputs

  def tiramisu(self, inputs):
    with tf.name_scope('Tiramisu'):
      concat_axis = Conv2dUtilities.channel_axis(inputs, self.data_format)
      number_of_sampling_steps = len(self.number_of_filters_for_convolution_blocks) - 1
      downsampling_tensors = []
      
      # Preprocessing convolution
      with tf.name_scope('Preprocessing'):
        inputs = tf.layers.conv2d(
            inputs=inputs, filters=self.number_of_preprocessing_convolution_filters, kernel_size=(3, 3), padding='same',
            activation=self.activation_function, data_format=self.data_format)
      
      # Downsampling
      with tf.name_scope('downsampling'):
        for i in range(number_of_sampling_steps):
          index = i
          number_of_filters = self.number_of_filters_for_convolution_blocks[index]
          inputs = self.__convolution_block(inputs, number_of_filters, 'downsampling_' + str(index + 1))
          
          downsampling_tensors.append(inputs)
          inputs = self.__downsample(inputs)
      
      # Upsampling
      with tf.name_scope('upsampling'):
        for i in range(number_of_sampling_steps):
          index = number_of_sampling_steps - i
          number_of_filters = self.number_of_filters_for_convolution_blocks[index]
          inputs = self.__convolution_block(inputs, number_of_filters, 'upsampling_' + str(index + 1))
          
          inputs = self.__upsample(inputs, self.number_of_filters_for_convolution_blocks[index - 1])
          
          downsampled_tensor = downsampling_tensors[index - 1]
          inputs = tf.concat([downsampled_tensor, inputs], concat_axis)
        
        inputs = self.__convolution_block(
            inputs, self.number_of_filters_for_convolution_blocks[0], 'upsampling_1')
      
      # Finalize the output to have the required number of channels.
      with tf.name_scope('Postprocessing'):
        inputs = tf.layers.conv2d(
            inputs=inputs, filters=self.number_of_output_filters, kernel_size=(1, 1), padding='same',
            activation=None, data_format=self.data_format)
    
    return inputs
