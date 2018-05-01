import tensorflow as tf
import Conv2dUtilities
import ChannelWeighting

class RefinementNet:

  def __init__(self, number_of_repetitions=3, number_of_blocks=4, number_of_convolutions_per_block=2, number_block_repetitions=2, number_of_filters_per_convolution=32, activation_function=tf.nn.relu, use_zero_padding=True, use_channel_weighting=False):
    self.number_of_repetitions = number_of_repetitions
    self.number_of_blocks = number_of_blocks
    self.number_of_convolutions_per_block = number_of_convolutions_per_block
    self.number_block_repetitions = number_block_repetitions
    self.number_of_filters_per_convolution = number_of_filters_per_convolution
    self.activation_function = activation_function
    self.use_zero_padding = use_zero_padding
    self.use_channel_weighting = use_channel_weighting
    
  def __convolution_block(self, inputs, number_of_output_filters, convolution_block_index, reuse, data_format):
    concat_axis = Conv2dUtilities.channel_axis(inputs, data_format)
  
    # TODO: Dilation should also produce additional channels!
  
    with tf.name_scope('dense_block'):
      for i in range(self.number_of_convolutions_per_block):
        with tf.name_scope('convolution_' + str(i + 1) + '_' + str(convolution_block_index)):
        
          # HACK: Number of dilations is hard coded for now
        
          use_dilation = False
          if use_dilation:
            number_of_dilation_1 = self.number_of_filters_per_convolution // 4
            number_of_dilation_2 = number_of_dilation_1 // 4
            number_of_dilation_1 = number_of_dilation_1 - number_of_dilation_2
            
            number_of_filters_per_convolution = self.number_of_filters_per_convolution - number_of_dilation_1 - number_of_dilation_2
            
            new_channels = Conv2dUtilities.convolution2d(inputs=inputs, filters=number_of_filters_per_convolution, kernel_size=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format, name='convolution_' + str(i + 1) + '_' + str(convolution_block_index), reuse=reuse)
            new_dilation_1_channels = Conv2dUtilities.convolution2d(inputs=inputs, filters=number_of_dilation_1, kernel_size=(3, 3), dilation_rate=(2, 2), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format, name='dilated_convolution_1_' + str(i + 1) + '_' + str(convolution_block_index), reuse=reuse)
            new_dilation_2_channels = Conv2dUtilities.convolution2d(inputs=inputs, filters=number_of_dilation_2, kernel_size=(3, 3), dilation_rate=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format, name='dilated_convolution_2_' + str(i + 1) + '_' + str(convolution_block_index), reuse=reuse)

            inputs = tf.concat([inputs, new_channels, new_dilation_1_channels, new_dilation_2_channels], concat_axis)
          
          else:
            if self.use_channel_weighting:
              inputs = ChannelWeighting.learned_channel_weighting(inputs, data_format, kernel_size=3, number_of_convolutions=2)
            new_channels = Conv2dUtilities.convolution2d(inputs=inputs, filters=self.number_of_filters_per_convolution, kernel_size=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format, name='convolution_' + str(i + 1) + '_' + str(convolution_block_index), reuse=reuse)
            inputs = tf.concat([inputs, new_channels], concat_axis)

    if self.use_channel_weighting:
      inputs = ChannelWeighting.learned_channel_weighting(inputs, data_format, kernel_size=3, number_of_convolutions=2)
    inputs = Conv2dUtilities.convolution2d(inputs=inputs, filters=number_of_output_filters, kernel_size=(3, 3), use_zero_padding=self.use_zero_padding, activation=self.activation_function, data_format=data_format, name='conv_final_' + str(convolution_block_index), reuse=reuse)
    return inputs

  def refinement_net(self, prediction_inputs, auxiliary_inputs, data_format='channels_last'):
    # TODO: Try without intermediate prediction and store the result directly in the prediction instead
  
    concat_axis = Conv2dUtilities.channel_axis(prediction_inputs, data_format)
    
    intermediate_prediction = prediction_inputs
    number_of_output_channels = intermediate_prediction.shape[concat_axis]
    
    with tf.name_scope('RefinementNet'):
      reuse = False
      for i in range(1 + self.number_of_repetitions):
        for j in range(self.number_of_blocks):
          reuse_block = False
          for k in range(1 + self.number_block_repetitions):
            inputs = tf.concat([intermediate_prediction, prediction_inputs, auxiliary_inputs], concat_axis)
            intermediate_prediction_delta = self.__convolution_block(inputs, number_of_output_channels, j + 1, reuse or reuse_block, data_format)
            intermediate_prediction = tf.add(intermediate_prediction, intermediate_prediction_delta)
            reuse_block = True
        reuse = True
    
    return intermediate_prediction