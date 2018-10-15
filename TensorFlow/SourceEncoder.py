from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Conv2dUtilities import Conv2dUtilities
from FeatureFlags import FeatureFlagMode
from Naming import Naming
from RenderPasses import RenderPasses

class SourceEncoder:

  # Remark: The source encoder is not like the one mentioned in the publication. For now it just prepares the data.

  def __init__(
      self, feature_prediction_tuple_type, auxiliary_features,
      feature_flags, feature_flag_mode, 
      number_of_output_channels, activation_function=tf.nn.relu,
      source_data_format='channels_last', data_format='channels_first'):
    self.feature_prediction_tuple_type = feature_prediction_tuple_type
    self.auxiliary_features = auxiliary_features
    self.feature_flags = feature_flags
    self.feature_flag_mode = feature_flag_mode
    self.number_of_output_channels = number_of_output_channels
    self.activation_function = activation_function
    self.source_data_format = source_data_format
    self.data_format = data_format
  
  def prepare_neural_network_input(self, feature_prediction_tuple, all_features):

    # We naively assume that this exists.
    source_concat_axis = Conv2dUtilities.channel_axis(
        feature_prediction_tuple.feature_predictions[0].source[0], self.source_data_format)

    # Prepare all the features we need.
    features = []
    for feature_prediction in feature_prediction_tuple.feature_predictions:
      features.append(feature_prediction)
    for feature in self.auxiliary_features:
      features.append(feature)
    
    # Merge the features into a tensor.
    result = []
    for index in range(len(feature_prediction_tuple.feature_predictions[0].source)):
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
    
    # Feature flags
    if (self.feature_flag_mode == FeatureFlagMode.ONE_HOT_ENCODING):
      feature_flags_name = Naming.feature_flags_name(feature_prediction_tuple.name)
      feature_flags = all_features[feature_flags_name]
      result = tf.concat([result, feature_flags], source_concat_axis)
    
    elif self.feature_flag_mode == FeatureFlagMode.EMBEDDING:
      # Adding the prediction feature flags does only work when the tensor is unbatched. This can be achieved with 'map_fn'.
      def add_feature_prediction_flags(inputs):
        local_concat_axis = Conv2dUtilities.channel_axis(inputs, self.source_data_format)
        height, width = Conv2dUtilities.height_width(inputs, self.source_data_format)
        feature_prediction_flags = self.feature_flags.feature_flags(
            feature_prediction_tuple.name, height, width, self.source_data_format)
        inputs = tf.concat([inputs, feature_prediction_flags], local_concat_axis)
        return inputs
      result = tf.map_fn(add_feature_prediction_flags, result)

    if self.data_format != self.source_data_format:
      result = Conv2dUtilities.convert_to_data_format(result, self.data_format)

    return result
  