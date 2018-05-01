import tensorflow as tf

def channel_axis(inputs, data_format):
  assert len(inputs.shape) == 3 or len(inputs.shape) == 4
  if len(inputs.shape) == 3:
    if data_format == 'channels_first':
      result = 0
    else:
      result = 2
  elif len(inputs.shape) == 4:
    if data_format == 'channels_first':
      result = 1
    else:
      result = 3
  return result

def number_of_channels(inputs, data_format):
  assert len(inputs.shape) == 4
  return inputs.shape[channel_axis(inputs, data_format)]

def non_zero_mask(inputs, data_format):
  channel_index = channel_axis(inputs, data_format)
  inputs = tf.abs(inputs)
  inputs = tf.reduce_sum(inputs, axis=channel_index)
  inputs = tf.sign(inputs)
  return inputs

def downsample(inputs, use_max_pooling, kernel_size=None, use_zero_padding=True, activation=None, data_format='channels_last', name=None):
  if use_max_pooling:
    return tf.layers.max_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2), padding='same', data_format=data_format, name=name)
  else:
    # TODO: Is this too general?
    return convolution2d(inputs, number_of_channels(inputs, data_format), kernel_size, use_zero_padding=use_zero_padding, strides=(2, 2), data_format=data_format, activation=activation, name=name)

def upsample(inputs, kernel_size, filters=None, activation=None, data_format='channels_last', name=None):
  if filters == None:
    filters = number_of_channels(inputs, data_format)
  return tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=activation, data_format=data_format, name=name)

def convolution2d(inputs, filters, kernel_size, use_zero_padding=True, strides=(1, 1), data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=None):
  if use_zero_padding:
    # Zero padding
    return tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding='same', data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, trainable=trainable, name=name, reuse=reuse)
  else:
    # Symmetric padding
    return __conv2d_utilities.conv2d_with_pad_mode(inputs, filters, kernel_size, strides=strides, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, trainable=trainable, name=name, reuse=reuse)

def __conv2d_with_pad_mode(inputs, filters, kernel_size, strides=(1, 1), mode='symmetric', constant_values=0, data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=None):
  
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  
  if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
    # Even kernel_size's are not yet supported
    raise NotImplementedError
  
  pad_height = kernel_size[0] // 2
  pad_width = kernel_size[1] // 2
  
  if dilation_rate != (1, 1):
    # Dilation is not yet supported
    print(dilation_rate)
    raise NotImplementedError
  
  if data_format == 'channels_first':
    # NCHW
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_height, pad_height], [pad_width, pad_width]], mode)
  else:
    # NHWC
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_height, pad_height], [pad_width, pad_width], [0, 0]], mode)
  
  padding='valid'
  inputs = tf.layers.conv2d(padded_inputs, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, trainable, name, reuse)
  
  return inputs