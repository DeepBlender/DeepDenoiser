#
# Based on: https://arxiv.org/abs/1505.04597
# 
# We are not using cropping which is one of the main differences to the original paper.
#

import tensorflow as tf
import Conv2dUtilities

# TODO: Consider batch normalization, dropout, regularization

class UNet:

  def __init__(self, number_of_initial_convolution_channels=64, number_of_sampling_steps=4, sampling_filter_multiplier=2, number_of_convolutions_per_block=2, activation_function=tf.nn.relu, use_zero_padding=True, use_max_pooling=True):
    self.number_of_initial_convolution_channels = number_of_initial_convolution_channels
    self.number_of_sampling_steps = number_of_sampling_steps
    self.sampling_filter_multiplier = sampling_filter_multiplier
    self.number_of_convolutions_per_block = number_of_convolutions_per_block
    self.activation_function = activation_function
    self.use_zero_padding = use_zero_padding
    self.use_max_pooling = use_max_pooling
    
  def __convolution_block(self, inputs, number_of_filters, data_format):
    with tf.name_scope('convolution_block'):
      for i in range(self.number_of_convolutions_per_block):
        with tf.name_scope('convolution_' + str(i + 1)):
          inputs = Conv2dUtilities.convolution2d(inputs=inputs, filters=number_of_filters, kernel_size=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format)
    return inputs

  def __downsample(self, inputs, data_format):
    with tf.name_scope('downsample'):
      return Conv2dUtilities.downsample(inputs=inputs, use_max_pooling=self.use_max_pooling, kernel_size=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format)

  def __upsample(self, inputs, number_of_filters, data_format):
    with tf.name_scope('upsample'):
      return Conv2dUtilities.upsample(inputs=inputs, kernel_size=(3, 3), filters=number_of_filters, activation=self.activation_function, data_format=data_format)

  def u_net(self, prediction_inputs, auxiliary_inputs, data_format='channels_last'):
    concat_axis = Conv2dUtilities.channel_axis(prediction_inputs, data_format)

    inputs = tf.concat([prediction_inputs, auxiliary_inputs], concat_axis)
    
    with tf.name_scope('U-Net'):
      downsampling_tensors = []
      
      with tf.name_scope('downsampling'):
        # Downsampling
        for i in range(self.number_of_sampling_steps):
          number_of_channels = (self.sampling_filter_multiplier ** i) * self.number_of_initial_convolution_channels
          inputs = self.__convolution_block(inputs, number_of_channels, data_format)
          downsampling_tensors.append(inputs)
          inputs = self.__downsample(inputs, data_format)
      
      with tf.name_scope('upsampling'):
        # Upsampling
        for i in range(self.number_of_sampling_steps):
          number_of_channels = (2 ** (self.number_of_sampling_steps - i)) * self.number_of_initial_convolution_channels
          inputs = self.__convolution_block(inputs, number_of_channels, data_format)
          inputs = self.__upsample(inputs, number_of_channels // 2, data_format)
          
          downsampled_tensor = downsampling_tensors[self.number_of_sampling_steps - 1 - i]
          inputs = tf.concat([downsampled_tensor, inputs], concat_axis)
      
      inputs = self.__convolution_block(inputs=inputs, number_of_filters=self.number_of_initial_convolution_channels, data_format=data_format)
      
    return inputs