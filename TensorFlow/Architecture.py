from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from enum import Enum

from RenderPasses import RenderPasses
from Naming import Naming

from FeatureEngineering import FeatureEngineering
from FeatureFlags import FeatureFlags
from FeatureFlags import FeatureFlagMode
from SourceEncoder import SourceEncoder
from KernelPrediction import KernelPrediction
from MultiScalePrediction import MultiScalePrediction

import Utilities
from Conv2dUtilities import Conv2dUtilities

from UNet import UNet
from Tiramisu import Tiramisu


class FeatureStandardization:

  def __init__(self, use_log1p, mean, variance, name):
    self.use_log1p = use_log1p
    self.mean = mean
    self.variance = variance
    self.name = name

  def use_mean(self):
    return self.mean != 0.
  
  def use_variance(self):
    return self.variance != 1.

  def standardize(self, feature, index):
    if self.use_log1p:
      feature = Utilities.signed_log1p(feature)
    if self.use_mean():
      feature = tf.subtract(feature, self.mean)
    if self.use_variance():
      feature = tf.divide(feature, tf.sqrt(self.variance))
    return feature
    
  def invert_standardization(self, feature):
    if self.use_variance():
      feature = tf.multiply(feature, tf.sqrt(self.variance))
    if self.use_mean():
      feature = tf.add(feature, self.mean)
    if self.use_log1p:
      feature = Utilities.signed_expm1(feature)
    return feature


class FeatureVariance:

  def __init__(self, use_variance, variance_mode, relative_variance, compute_before_standardization, compress_to_one_channel, name):
    self.use_variance = use_variance
    self.variance_mode = variance_mode
    self.relative_variance = relative_variance
    self.compute_before_standardization = compute_before_standardization
    self.compress_to_one_channel = compress_to_one_channel
    self.name = name
  
  def variance(self, inputs, epsilon=1e-4, data_format='channels_last'):
    assert self.use_variance
    result = FeatureEngineering.variance(
        inputs, variance_mode=self.variance_mode, relative_variance=self.relative_variance, compress_to_one_channel=self.compress_to_one_channel,
        epsilon=epsilon, data_format=data_format)
    return result


class FeaturePredictionType(Enum):
  COLOR = 1
  DIRECT = 2
  INDIRECT = 3
  AUXILIARY = 4

class FeaturePrediction:

  def __init__(
      self, feature_prediction_type, load_data,
      number_of_sources, preserve_source, is_target,
      feature_standardization, invert_standardization, feature_variance,
      number_of_channels, name):
    
    self.feature_prediction_type = feature_prediction_type
    self.load_data = load_data
    self.number_of_sources = number_of_sources
    self.preserve_source = preserve_source
    self.is_target = is_target
    self.feature_standardization = feature_standardization
    self.invert_standardization = invert_standardization
    self.feature_variance = feature_variance
    self.number_of_channels = number_of_channels
    self.name = name
    self.predictions = []

  def initialize_sources_from_dictionary(self, dictionary):
    self.source = []
    self.variance = []
    if self.preserve_source:
      self.preserved_source = []

    for index in range(self.number_of_sources):
      source = dictionary[Naming.source_feature_name(self.name, index=index)]
      self.source.append(source)
      if self.preserve_source:
        self.preserved_source.append(source)

  def standardize(self):
    with tf.name_scope(Naming.tensorboard_name(self.name + ' Variance')):
      if self.feature_variance.use_variance and self.feature_variance.compute_before_standardization:
        for index in range(self.number_of_sources):
          assert len(self.variance) == index
          variance = self.feature_variance.variance(self.source[index], data_format='channels_last')
          self.variance.append(variance)
    
    with tf.name_scope(Naming.tensorboard_name('Standardize ' + self.name)):
      if self.feature_standardization != None:
        for index in range(self.number_of_sources):
          self.source[index] = self.feature_standardization.standardize(self.source[index], index)
    
    with tf.name_scope(Naming.tensorboard_name(self.name + ' Variance')):
      if self.feature_variance.use_variance and not self.feature_variance.compute_before_standardization:
        for index in range(self.number_of_sources):
          assert len(self.variance) == index
          variance = self.feature_variance.variance(self.source[index], data_format='channels_last')
          self.variance.append(variance)
  
  def prediction_invert_standardization(self):
    with tf.name_scope(Naming.tensorboard_name('Invert Standardization ' + self.name)):
      if self.feature_standardization != None:
        for index in range(len(self.predictions)):
          self.predictions[index] = self.feature_standardization.invert_standardization(self.predictions[index])
  
  def add_prediction(self, scale_index, prediction):
    if not self.is_target:
      raise Exception('Adding a prediction for a feature that is not a target is not allowed.')
    while len(self.predictions) <= scale_index:
      self.predictions.append(None)
    self.predictions[scale_index] = prediction
  
  def add_prediction_to_dictionary(self, scale_index, dictionary):
    if self.is_target:
      if self.load_data:
        prediction = self.predictions[scale_index]
      else:
        # If the data was generated, we don't care about the prediction.
        # Let's make sure it has not negative impact during the training.
        prediction = self.predictions[scale_index]
        start = [0, 0, 0, 0]
        shape = tf.shape(prediction)
        prediction = tf.slice(self.source[0], start, shape)
      
      # Make sure the prediction has the correct amount of channels.
      if Conv2dUtilities.number_of_channels(prediction, 'channels_last') != self.number_of_channels:
        assert self.number_of_channels == 1
        channel_axis = Conv2dUtilities.channel_axis(prediction, 'channels_last')
        prediction, _ = tf.split(prediction, [1, 2], channel_axis)

      dictionary[Naming.feature_prediction_name(self.name)] = prediction

  @staticmethod
  def feature_prediction_type_to_string(feature_prediction_type):
    result = ''
    if feature_prediction_type == FeaturePredictionType.COLOR:
      result = 'Color'
    elif feature_prediction_type == FeaturePredictionType.DIRECT:
      result = 'Direct'
    elif feature_prediction_type == FeaturePredictionType.INDIRECT:
      result = 'Indirect'
    elif feature_prediction_type == FeaturePredictionType.AUXILIARY:
      result = 'Auxiliary'
    return result


class FeaturePredictionTupleType(Enum):
  SINGLE = 1
  COMBINED = 2

class FeaturePredictionTuple:

  def __init__(
      self, feature_predictions, feature_prediction_tuple_type, name):
    self.feature_predictions = feature_predictions
    self.feature_prediction_tuple_type = feature_prediction_tuple_type
    self.name = name


class CoreArchitecture:

  def __init__(
      self, name='U-Net', number_of_filters_for_convolution_blocks=[128, 128, 128], number_of_convolutions_per_block=5,
      activation_function=tf.nn.relu, use_batch_normalization=False, dropout_rate=0., use_multiscale_predictions=True,
      data_format='channels_first'):
    self.name = name
    
    if self.name == 'U-Net':
      self.architecture = UNet(
          number_of_filters_for_convolution_blocks=number_of_filters_for_convolution_blocks,
          number_of_convolutions_per_block=number_of_convolutions_per_block,
          use_multiscale_output=use_multiscale_predictions,
          activation_function=activation_function,
          use_batch_normalization=use_batch_normalization,
          dropout_rate=dropout_rate,
          data_format=data_format)
    else:
      assert self.name == 'Tiramisu'
      self.architecture = Tiramisu(
          # TODO: Make it configurable (DeepBlender)
          number_of_preprocessing_convolution_filters=number_of_filters_for_convolution_blocks[0],
          
          number_of_filters_for_convolution_blocks=number_of_filters_for_convolution_blocks,
          number_of_convolutions_per_block=number_of_convolutions_per_block,
          use_multiscale_output=use_multiscale_predictions,
          activation_function=activation_function,
          use_batch_normalization=use_batch_normalization,
          dropout_rate=dropout_rate,
          data_format=data_format)
  
  def predict(self, inputs, is_training):
    inputs = self.architecture.predict(inputs, is_training)
    return inputs


class AdjustNumberOfChannels:

  def __init__(self, number_of_output_channels, activation_function=tf.nn.relu, data_format='channels_first'):
    self.number_of_output_channels = number_of_output_channels
    self.activation_function = activation_function
    self.data_format = data_format
  
  def predict(self, inputs):
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=self.number_of_output_channels, kernel_size=(1, 1), padding='same',
        activation=self.activation_function, data_format=self.data_format)
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=self.number_of_output_channels, kernel_size=(1, 1), padding='same',
        activation=None, data_format=self.data_format)
    return inputs


class KernelPredictor:

  # TODO: Consider number_of_sources_per_target and use swa over all of them.

  def __init__(
      self, use_kernel_prediction, kernel_size, use_standardized_source_for_kernel_prediction,
      source_data_format='channels_last', data_format='channels_first'):
    self.use_kernel_prediction = use_kernel_prediction
    self.kernel_size = kernel_size
    self.use_standardized_source_for_kernel_prediction = use_standardized_source_for_kernel_prediction
    self.source_data_format = source_data_format
    self.data_format = data_format

  def predict(self, feature_prediction):
    if self.use_kernel_prediction:
      if self.use_standardized_source_for_kernel_prediction:
        source = feature_prediction.source[0]
      else:
        source = feature_prediction.preserved_source[0]
      
      if self.data_format != self.source_data_format:
        source = Conv2dUtilities.convert_to_data_format(source, self.data_format)

      channel_axis = Conv2dUtilities.channel_axis(source, self.data_format)

      # Ensure we always have 3 channels as input.
      if Conv2dUtilities.number_of_channels(source, self.data_format) != 3:
        assert Conv2dUtilities.number_of_channels(source, self.data_format) == 1
        source = tf.concat([source, source, source], channel_axis)

      for scale_index in range(len(feature_prediction.predictions)):
        kernel = feature_prediction.predictions[scale_index]

        scaled_source = source
        if scale_index > 0:
          size = 2 ** scale_index
          scaled_source = MultiScalePrediction.scale_down(scaled_source, heigh_width_scale_factor=size, data_format=self.data_format)
        with tf.name_scope(Naming.tensorboard_name('kernel_prediction_' + feature_prediction.name)):
          
          prediction = KernelPrediction.kernel_prediction(
              scaled_source, kernel,
              self.kernel_size, data_format=self.data_format)
        feature_prediction.add_prediction(scale_index, prediction)


class MultiScalePredictor:

  def __init__(
      self, use_multiscale_predictions, invert_standardization_after_multiscale_predictions,
      source_data_format='channels_last', data_format='channels_first'):
    self.use_multiscale_predictions = use_multiscale_predictions
    self.invert_standardization_after_multiscale_predictions = invert_standardization_after_multiscale_predictions
    self.source_data_format = source_data_format
    self.data_format = data_format
  
  def predict(self, feature_prediction, multiscale_combine_reuse):
    if not self.invert_standardization_after_multiscale_predictions:
      with tf.name_scope('invert_standardization'):
        if feature_prediction.invert_standardization:
          feature_prediction.prediction_invert_standardization()
    
    with tf.name_scope('combine_multiscales'):
      if self.use_multiscale_predictions:
        for scale_index in range(len(feature_prediction.predictions) - 1, 0, -1):
          larger_scale_index = scale_index - 1
          
          small_prediction = feature_prediction.predictions[scale_index]
          prediction = feature_prediction.predictions[larger_scale_index]
          
          with tf.variable_scope('reused_compose_scales', reuse=multiscale_combine_reuse):
            prediction = MultiScalePrediction.compose_scales(small_prediction, prediction, data_format=self.data_format)
          multiscale_combine_reuse = True
          
          feature_prediction.add_prediction(larger_scale_index, prediction)
    
    if self.invert_standardization_after_multiscale_predictions:
      with tf.name_scope('invert_standardization'):
        if feature_prediction.invert_standardization:
          feature_prediction.prediction_invert_standardization()


class DataFormatReverter:

  def __init__(self, source_data_format='channels_last', data_format='channels_first'):
    self.source_data_format = source_data_format
    self.data_format = data_format
    
  def predict(self, feature_prediction):
    # Convert back to the source data format if needed.
    if self.data_format != self.source_data_format:
      for scale_index in range(len(feature_prediction.predictions)):
        feature_prediction.predictions[scale_index] = Conv2dUtilities.convert_to_data_format(feature_prediction.predictions[scale_index], self.source_data_format)


class Architecture:

  def __init__(self, parsed_json, source_data_format='channels_last', data_format='channels_first'):
    self.source_data_format = source_data_format
    self.data_format = data_format
    
    self.model_directory = parsed_json['model_directory']
    self.number_of_sources_per_target = parsed_json['number_of_sources_per_target']
    
    architecture_json = parsed_json['architecture']
    combined_features_json = parsed_json['combined_features']
    combined_features_handling_json = parsed_json['combined_features_handling']
    auxiliary_features_json = parsed_json['auxiliary_features']

    self.feature_prediction_tuple_type = architecture_json['source_encoder']['feature_prediction_tuple_type']
    self.feature_prediction_tuple_type = FeaturePredictionTupleType[self.feature_prediction_tuple_type]

    self.__preserve_source = not architecture_json['kernel_prediction']['use_standardized_source_for_kernel_prediction']
    self.__number_of_core_architecture_input_channels = architecture_json['core_architecture']['number_of_filters_for_convolution_blocks'][0]
    
    # Requires 'self.number_of_sources_per_target' and 'self.__preserve_source'
    self.__prepare_feature_predictions(combined_features_json, combined_features_handling_json, auxiliary_features_json)
    
    # Requires 'self.feature_predictions', 'self.__number_of_core_architecture_input_channels', 'self.source_data_format' and 'self.data_format'
    self.__prepare_architecture(architecture_json, combined_features_json)
  
  def __prepare_feature_predictions(self, combined_features_json, combined_features_handling_json, auxiliary_features_json):
  
    # The auxiliary names have to be sorted, to ensure they are not randomly mixed between runs.
    auxiliary_feature_names = sorted(list(auxiliary_features_json.keys()))
    self.auxiliary_features = []
    for feature_name in auxiliary_feature_names:
      feature = auxiliary_features_json[feature_name]
      
      feature_variance = feature['feature_variance']
      feature_variance = FeatureVariance(
          feature_variance['use_variance'], feature_variance['variance_mode'], feature_variance['relative_variance'],
          feature_variance['compute_before_standardization'], feature_variance['compress_to_one_channel'],
          feature_name)
      feature_standardization = feature['standardization']
      feature_standardization = FeatureStandardization(
          feature_standardization['use_log1p'], feature_standardization['mean'], feature_standardization['variance'],
          feature_name)
      is_target = False
      invert_standardization = False
      auxiliary_feature = FeaturePrediction(
          FeaturePredictionType.AUXILIARY, True,
          self.number_of_sources_per_target, self.__preserve_source, is_target,
          feature_standardization, invert_standardization, feature_variance,
          feature['number_of_channels'], feature_name)
      self.auxiliary_features.append(auxiliary_feature)  
    

    # Prepare how features should be handled.
    feature_prediction_type_to_feature_variance = {}
    feature_prediction_type_to_feature_standardization = {}
    feature_prediction_type_to_invert_standardization = {}
    for feature_type in [FeaturePredictionType.COLOR, FeaturePredictionType.DIRECT, FeaturePredictionType.INDIRECT]:
      feature_handling = combined_features_handling_json[FeaturePrediction.feature_prediction_type_to_string(feature_type)]
      feature_variance = feature_handling['feature_variance']
      feature_variance = FeatureVariance(
          feature_variance['use_variance'], feature_variance['variance_mode'], feature_variance['relative_variance'],
          feature_variance['compute_before_standardization'], feature_variance['compress_to_one_channel'],
          feature_name)
      feature_standardization = feature_handling['standardization']
      feature_standardization = FeatureStandardization(
          feature_standardization['use_log1p'], feature_standardization['mean'], feature_standardization['variance'],
          feature_name)
      invert_standardization = feature_handling['invert_standardization']

      feature_prediction_type_to_feature_variance[feature_type] = feature_variance
      feature_prediction_type_to_feature_standardization[feature_type] = feature_standardization
      feature_prediction_type_to_invert_standardization[feature_type] = invert_standardization

    
    # Combined feature predictions

    self.feature_predictions = []
    self.feature_prediction_tuples = []
    combined_feature_names = sorted(list(combined_features_json.keys()))

    for combined_feature_name in combined_feature_names:
      color_feature_prediction = None
      direct_feature_prediction = None
      indirect_feature_prediction = None
      
      combined_feature = combined_features_json[combined_feature_name]

      for feature_type in [FeaturePredictionType.COLOR, FeaturePredictionType.DIRECT, FeaturePredictionType.INDIRECT]:
        is_target = True
        feature_name = combined_feature[FeaturePrediction.feature_prediction_type_to_string(feature_type)]

        feature_standardization = feature_prediction_type_to_feature_standardization[feature_type]
        invert_standardization = feature_prediction_type_to_invert_standardization[feature_type]
        feature_variance = feature_prediction_type_to_feature_variance[feature_type]
        number_of_channels = RenderPasses.number_of_channels(feature_name)
        load_data = True

        if feature_name == None or feature_name == '':
          feature_name = combined_feature_name + ' ' + FeaturePrediction.feature_prediction_type_to_string(feature_type)
          load_data = False

        if load_data or self.feature_prediction_tuple_type == FeaturePredictionTupleType.COMBINED:
          feature_prediction = FeaturePrediction(
              feature_type, load_data,
              self.number_of_sources_per_target, self.__preserve_source, is_target,
              feature_standardization, invert_standardization, feature_variance,
              number_of_channels, feature_name)
          self.feature_predictions.append(feature_prediction)
        else:
          feature_prediction = None

        if feature_type == FeaturePredictionType.COLOR:
          color_feature_prediction = feature_prediction
        elif feature_type == FeaturePredictionType.DIRECT:
          direct_feature_prediction = feature_prediction
        elif feature_type == FeaturePredictionType.INDIRECT:
          indirect_feature_prediction = feature_prediction

      if self.feature_prediction_tuple_type == FeaturePredictionTupleType.COMBINED:
        feature_prediction_tuple = FeaturePredictionTuple(
            [color_feature_prediction, direct_feature_prediction, indirect_feature_prediction],
            self.feature_prediction_tuple_type,
            combined_feature_name)
        self.feature_prediction_tuples.append(feature_prediction_tuple)

    if self.feature_prediction_tuple_type == FeaturePredictionTupleType.SINGLE:
      for feature_prediction in self.feature_predictions:
        feature_prediction_tuple = FeaturePredictionTuple(
            [feature_prediction],
            self.feature_prediction_tuple_type,
            feature_prediction.name)
        self.feature_prediction_tuples.append(feature_prediction_tuple)
  
  def __prepare_architecture(self, architecture_json, combined_features_json):
    source_encoder_json = architecture_json['source_encoder']
    core_architecture_json = architecture_json['core_architecture']
    kernel_prediction_json = architecture_json['kernel_prediction']
    multiscale_prediction_json = architecture_json['multiscale_prediction']
    
    self.use_kernel_prediction = kernel_prediction_json['use_kernel_prediction']
    self.use_multiscale_predictions = multiscale_prediction_json['use_multiscale_predictions']
    
    feature_flags = []
    for feature_prediction_tuple in self.feature_prediction_tuples:
      feature_flags.append(feature_prediction_tuple.name)

    self.feature_flags = FeatureFlags(
        feature_flags, FeatureFlagMode[source_encoder_json['feature_flag_mode']],
        'channels_last')

    feature_flag_mode = self.feature_flags.feature_flag_mode
    if feature_flag_mode != FeatureFlagMode.EMBEDDING:
      self.feature_flags = None

    self.source_encoder = SourceEncoder(
        self.feature_prediction_tuple_type,
        self.auxiliary_features, self.feature_flags, feature_flag_mode,
        self.__number_of_core_architecture_input_channels, activation_function=tf.nn.relu,
        source_data_format=self.source_data_format, data_format=self.data_format)
    
    self.core_architecture = CoreArchitecture(
        name=core_architecture_json['name'],
        number_of_filters_for_convolution_blocks=core_architecture_json['number_of_filters_for_convolution_blocks'],
        number_of_convolutions_per_block=core_architecture_json['number_of_convolutions_per_block'],
        activation_function=tf.nn.relu, use_batch_normalization=False, dropout_rate=0.,
        use_multiscale_predictions=self.use_multiscale_predictions,
        data_format=self.data_format)
    
    if self.feature_prediction_tuple_type == FeaturePredictionTupleType.SINGLE:
      feature_prediction_tuple_size = 1
    else:
      feature_prediction_tuple_size = 3
    
    if self.use_kernel_prediction:
      # We predict feature prediction tuple size features each for kernel_size ^ 2.
      # This is done over all sources.
      number_of_output_channels = (
          self.number_of_sources_per_target * feature_prediction_tuple_size * (kernel_prediction_json['kernel_size'] ** 2))
    else:
      # The number of actual output channels is 3 for each feature we are predicting.
      number_of_output_channels = feature_prediction_tuple_size * 3

    self.core_architecture_postprocess = AdjustNumberOfChannels(
        number_of_output_channels, activation_function=tf.nn.relu, data_format=self.data_format)

    self.kernel_predictor = KernelPredictor(
        self.use_kernel_prediction, kernel_prediction_json['kernel_size'], kernel_prediction_json['use_standardized_source_for_kernel_prediction'],
        source_data_format=self.source_data_format, data_format=self.data_format)
    
    self.multiscale_predictor = MultiScalePredictor(
        self.use_multiscale_predictions, multiscale_prediction_json['invert_standardization_after_multiscale_predictions'],
        source_data_format=self.source_data_format, data_format=self.data_format)
    
    self.data_format_reverter = DataFormatReverter(source_data_format=self.source_data_format, data_format=self.data_format)
    
  def predict(self, features, mode):
    is_training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
      is_training = True

    # Initialize the prediction features' sources
    for feature_prediction in self.feature_predictions:
      feature_prediction.initialize_sources_from_dictionary(features)
    
    for auxiliary_feature in self.auxiliary_features:
      auxiliary_feature.initialize_sources_from_dictionary(features)

    # Standardization of the data
    with tf.name_scope('standardize'):
      for feature_prediction in self.feature_predictions:
        feature_prediction.standardize()
      
      for auxiliary_feature in self.auxiliary_features:
        auxiliary_feature.standardize()
    
    with tf.name_scope('feature_predictions'):
      reuse_core_architecture = False
      multiscale_combine_reuse = False

      for feature_prediction_tuple in self.feature_prediction_tuples:

        with tf.name_scope('prepare_network_input'):
          inputs = self.source_encoder.prepare_neural_network_input(feature_prediction_tuple, features)

        with tf.name_scope('core_architecture'):
          with tf.variable_scope('reused_core_architecture', reuse=reuse_core_architecture):
            inputs = self.core_architecture.predict(inputs, is_training)
            
            # Reuse the variables after the first pass.
            reuse_core_architecture = True
            
            with tf.name_scope('Postprocess'):
              for index in range(len(inputs)):
                inputs[index] = self.core_architecture_postprocess.predict(inputs[index])

            if self.use_multiscale_predictions:
              # Reverse the inputs, such that it is sorted from largest to smallest.
              inputs = list(reversed(inputs))
          
          channel_axis = Conv2dUtilities.channel_axis(inputs[0], self.data_format)
          for scale_index in range(len(inputs)):
            predictions = tf.split(inputs[scale_index], len(feature_prediction_tuple.feature_predictions), channel_axis)

            for index, prediction in enumerate(predictions):
              feature_prediction = feature_prediction_tuple.feature_predictions[index]
              feature_prediction.add_prediction(scale_index, prediction)
          
          with tf.name_scope(Naming.tensorboard_name('kernel_prediction_' + feature_prediction_tuple.name)):
            for feature_prediction in feature_prediction_tuple.feature_predictions:
              self.kernel_predictor.predict(feature_prediction)
          
          for feature_prediction in feature_prediction_tuple.feature_predictions:
            self.multiscale_predictor.predict(feature_prediction, multiscale_combine_reuse)
            multiscale_combine_reuse = True
          
          # Convert back to the source data format if needed.
          with tf.name_scope('revert_data_format_conversion'):
            for feature_prediction in feature_prediction_tuple.feature_predictions:
              self.data_format_reverter.predict(feature_prediction)

    # Create the prediction dictionaries to be returned
    target_feature_prediction = None
    for feature_prediction in self.feature_predictions:
      if feature_prediction.is_target:
        target_feature_prediction = feature_prediction
        break
    
    prediction_dictionaries = []
    for scale_index in range(len(target_feature_prediction.predictions)):
      prediction_dictionary = {}
      for feature_prediction in self.feature_predictions:
        if feature_prediction.is_target:
          feature_prediction.add_prediction_to_dictionary(scale_index, prediction_dictionary)
      prediction_dictionaries.append(prediction_dictionary)
      
    return prediction_dictionaries
