from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Conv2dUtilities import Conv2dUtilities

class SourceEncoder:

  # Remark: The source encoder is not like the one mentioned in the publication. For now it just prepares the data.

  def __init__(
      self, prediction_features, feature_flags, use_all_targets_as_input, number_of_output_channels,
      activation_function=tf.nn.relu, source_data_format='channels_last', data_format='channels_first'):
    self.feature_flags = feature_flags
    self.use_all_targets_as_input = use_all_targets_as_input
    self.number_of_output_channels = number_of_output_channels
    self.activation_function = activation_function
    self.source_data_format = source_data_format
    self.data_format = data_format
  
    self.source_features = []
    self.target_features = []
    for feature in prediction_features:
      if feature.is_target:
        self.target_features.append(feature)
      else:
        self.source_features.append(feature)
  
  def prepare_neural_network_input(self, prediction_feature):
    source_concat_axis = Conv2dUtilities.channel_axis(prediction_feature.source[0], self.source_data_format)

    # Prepare all the features we need.
    features = []
    if self.use_all_targets_as_input:
      for feature in self.target_features:
        features.append(feature)
    else:
      features.append(prediction_feature)
    
    for feature in self.source_features:
      features.append(feature)
    
    
    # Merge the features into a tensor.
    result = []
    for index in range(len(prediction_feature.source)):
      for feature in features:
        source = feature.source[index]

        # Each input feature needs exactly three channels.
        if Conv2dUtilities.number_of_channels(source, source_concat_axis) != 3:
          assert Conv2dUtilities.number_of_channels(source, source_concat_axis) == 1
          source = tf.concat([source, source, source], source_concat_axis)

        result.append(source)
        if feature.feature_variance.use_variance:
          variance = feature.variance[index]
          result.append(variance)
    result = tf.concat(result, source_concat_axis)
    
    
    # Adding the prediction feature flags does only work when the tensor is unbatched. This can be achieved with 'map_fn'.
    def add_prediction_feature_flags(inputs):
      local_concat_axis = Conv2dUtilities.channel_axis(inputs, self.source_data_format)
      height, width = Conv2dUtilities.height_width(inputs, self.source_data_format)
      prediction_feature_flags = self.feature_flags.feature_flags(prediction_feature.name, height, width, self.source_data_format)
      inputs = tf.concat([inputs, prediction_feature_flags], local_concat_axis)
      return inputs
    result = tf.map_fn(add_prediction_feature_flags, result)

    
    if self.data_format != self.source_data_format:
      result = Conv2dUtilities.convert_to_data_format(result, self.data_format)

    return result
  