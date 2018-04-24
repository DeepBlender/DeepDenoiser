import tensorflow as tf
import Conv2dUtilities

def learned_channel_weighting(inputs, data_format, kernel_size=1, activation_function=tf.nn.relu, number_of_convolutions=2):
  
  # Inspired by squeeze and excite.
  
  with tf.name_scope('ChannelWeighting'):
  
    if data_format == 'channels_last':
      number_of_channels = inputs.shape[3]
      weights = tf.reduce_mean(inputs, axis=[1, 2])
      weights = tf.expand_dims(weights, 1)
      weights = tf.expand_dims(weights, 1)
    else:
      number_of_channels = inputs.shape[1]
      weights = tf.reduce_mean(inputs, axis=[2, 3])
      weights = tf.expand_dims(weights, -1)
      weights = tf.expand_dims(weights, -1)
    
    for i in range(number_of_convolutions):
      weights = Conv2dUtilities.convolution2d(
          inputs=weights,
          filters=number_of_channels,
          kernel_size=[kernel_size, kernel_size],
          activation=activation_function,
          data_format=data_format)
    
    inputs = tf.multiply(inputs, weights)
  return inputs