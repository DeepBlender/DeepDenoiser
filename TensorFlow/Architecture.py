from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from RenderPasses import RenderPasses
from Naming import Naming

from FeatureEngineering import FeatureEngineering
from FeatureFlags import FeatureFlags
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

class PredictionFeature:

  def __init__(
      self, number_of_sources, preserve_source, is_target, feature_standardization, invert_standardization, feature_variance,
      feature_flag_names, number_of_channels, name):
    
    self.number_of_sources = number_of_sources
    self.preserve_source = preserve_source
    self.is_target = is_target
    self.feature_standardization = feature_standardization
    self.invert_standardization = invert_standardization
    self.feature_variance = feature_variance
    self.feature_flag_names = feature_flag_names
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
      dictionary[Naming.prediction_feature_name(self.name)] = self.predictions[scale_index]

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

  def __init__(
      self, use_kernel_prediction, kernel_size, use_standardized_source_for_kernel_prediction,
      source_data_format='channels_last', data_format='channels_first'):
    self.use_kernel_prediction = use_kernel_prediction
    self.kernel_size = kernel_size
    self.use_standardized_source_for_kernel_prediction = use_standardized_source_for_kernel_prediction
    self.source_data_format = source_data_format
    self.data_format = data_format
  
  def predict(self, prediction_feature):
    if self.use_kernel_prediction:
      if self.use_standardized_source_for_kernel_prediction:
        source = prediction_feature.source[0]
      else:
        source = prediction_feature.preserved_source[0]
      
      if self.data_format != self.source_data_format:
        source = Conv2dUtilities.convert_to_data_format(source, self.data_format)

      for scale_index in range(len(prediction_feature.predictions)):
        scaled_source = source
        if scale_index > 0:
          size = 2 ** scale_index
          scaled_source = MultiScalePrediction.scale_down(scaled_source, heigh_width_scale_factor=size, data_format=self.data_format)
        with tf.name_scope(Naming.tensorboard_name('kernel_prediction_' + prediction_feature.name)):
          prediction = KernelPrediction.kernel_prediction(
              scaled_source, prediction_feature.predictions[scale_index],
              self.kernel_size, data_format=self.data_format)
        prediction_feature.add_prediction(scale_index, prediction)

class MultiScalePredictor:

  def __init__(
      self, use_multiscale_predictions, invert_standardization_after_multiscale_predictions,
      source_data_format='channels_last', data_format='channels_first'):
    self.use_multiscale_predictions = use_multiscale_predictions
    self.invert_standardization_after_multiscale_predictions = invert_standardization_after_multiscale_predictions
    self.source_data_format = source_data_format
    self.data_format = data_format
  
  def predict(self, prediction_feature, multiscale_combine_reuse):
    if not self.invert_standardization_after_multiscale_predictions:
      with tf.name_scope('invert_standardization'):
        if prediction_feature.invert_standardization:
          prediction_feature.prediction_invert_standardization()
    
    with tf.name_scope('combine_multiscales'):
      if self.use_multiscale_predictions:
        for scale_index in range(len(prediction_feature.predictions) - 1, 0, -1):
          larger_scale_index = scale_index - 1
          
          small_prediction = prediction_feature.predictions[scale_index]
          prediction = prediction_feature.predictions[larger_scale_index]
          
          with tf.variable_scope('reused_compose_scales', reuse=multiscale_combine_reuse):
            prediction = MultiScalePrediction.compose_scales(small_prediction, prediction, data_format=self.data_format)
          multiscale_combine_reuse = True
          
          prediction_feature.add_prediction(larger_scale_index, prediction)
    
    if self.invert_standardization_after_multiscale_predictions:
      with tf.name_scope('invert_standardization'):
        if prediction_feature.invert_standardization:
          prediction_feature.prediction_invert_standardization()
    
class DataFormatReverter:

  def __init__(self, source_data_format='channels_last', data_format='channels_first'):
    self.source_data_format = source_data_format
    self.data_format = data_format
    
  def predict(self, prediction_feature):
    # Convert back to the source data format if needed.
    if self.data_format != self.source_data_format:
      for scale_index in range(len(prediction_feature.predictions)):
        prediction_feature.predictions[scale_index] = Conv2dUtilities.convert_to_data_format(prediction_feature.predictions[scale_index], self.source_data_format)
    
class Architecture:

  def __init__(self, parsed_json, source_data_format='channels_last', data_format='channels_first'):
    self.source_data_format = source_data_format
    self.data_format = data_format
    
    self.model_directory = parsed_json['model_directory']
    self.number_of_sources_per_target = parsed_json['number_of_sources_per_target']
    architecture_json = parsed_json['architecture']
    features_json = parsed_json['features']
    
    self.__preserve_source = not architecture_json['kernel_prediction']['use_standardized_source_for_kernel_prediction']
    self.__number_of_core_architecture_input_channels = architecture_json['core_architecture']['number_of_filters_for_convolution_blocks'][0]
    
    # Requires 'self.number_of_sources_per_target' and 'self.__preserve_source'
    self.__prepare_prediction_features(features_json)
    
    # Requires 'self.prediction_features', 'self.__number_of_core_architecture_input_channels', 'self.source_data_format' and 'self.data_format'
    self.__prepare_architecture(architecture_json)
    
  def __prepare_prediction_features(self, features_json):
  
    # The names have to be sorted, otherwise the channels would be randomly mixed.
    feature_names = sorted(list(features_json.keys()))
    
    self.prediction_features = []
    for feature_name in feature_names:
      feature = features_json[feature_name]
      
      # REMARK: It is assumed that there are no features which are only a target, without also being a source.
      if feature['is_source']:
        feature_variance = feature['feature_variance']
        feature_variance = FeatureVariance(
            feature_variance['use_variance'], feature_variance['variance_mode'], feature_variance['relative_variance'],
            feature_variance['compute_before_standardization'], feature_variance['compress_to_one_channel'],
            feature_name)
        feature_standardization = feature['standardization']
        feature_standardization = FeatureStandardization(
            feature_standardization['use_log1p'], feature_standardization['mean'], feature_standardization['variance'],
            feature_name)
        invert_standardization = feature['invert_standardization']
        prediction_feature = PredictionFeature(
            self.number_of_sources_per_target, self.__preserve_source, feature['is_target'],
            feature_standardization, invert_standardization, feature_variance,
            feature['feature_flags'], feature['number_of_channels'], feature_name)
        self.prediction_features.append(prediction_feature)  

  def __prepare_architecture(self, architecture_json):
    source_encoder_json = architecture_json['source_encoder']
    core_architecture_json = architecture_json['core_architecture']
    kernel_prediction_json = architecture_json['kernel_prediction']
    multiscale_prediction_json = architecture_json['multiscale_prediction']
    
    self.use_kernel_prediction = kernel_prediction_json['use_kernel_prediction']
    self.use_multiscale_predictions = multiscale_prediction_json['use_multiscale_predictions']
    
    self.feature_flags = FeatureFlags(source_encoder_json['feature_flags'])
    for prediction_feature in self.prediction_features:
      if prediction_feature.is_target:
        self.feature_flags.add_render_pass_name_to_feature_flag_names(
            prediction_feature.name, prediction_feature.feature_flag_names)
    self.feature_flags.freeze()
    
    self.source_encoder = SourceEncoder(
        self.prediction_features, self.feature_flags, source_encoder_json['use_all_targets_as_input'],
        self.__number_of_core_architecture_input_channels, activation_function=tf.nn.relu,
        source_data_format=self.source_data_format, data_format=self.data_format)
    
    self.core_architecture = CoreArchitecture(
        name=core_architecture_json['name'],
        number_of_filters_for_convolution_blocks=core_architecture_json['number_of_filters_for_convolution_blocks'],
        number_of_convolutions_per_block=core_architecture_json['number_of_convolutions_per_block'],
        activation_function=tf.nn.relu, use_batch_normalization=False, dropout_rate=0.,
        use_multiscale_predictions=self.use_multiscale_predictions,
        data_format=self.data_format)
    
    if self.use_kernel_prediction:
      number_of_output_channels = kernel_prediction_json['kernel_size'] ** 2
    else:
      number_of_output_channels = prediction_features[0].number_of_channels
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
    for prediction_feature in self.prediction_features:
      prediction_feature.initialize_sources_from_dictionary(features)
    
    # Standardization of the data
    with tf.name_scope('standardize'):
      for prediction_feature in self.prediction_features:
        prediction_feature.standardize()
    
    with tf.name_scope('feature_predictions'):
      reuse_core_architecture = False
      multiscale_combine_reuse = False
      
      for prediction_feature in self.prediction_features:
        if prediction_feature.is_target:
          
          with tf.name_scope('prepare_network_input'):
            inputs = self.source_encoder.prepare_neural_network_input(prediction_feature)
          
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
          
          for scale_index in range(len(inputs)):
            prediction_feature.add_prediction(scale_index, inputs[scale_index])
          
          with tf.name_scope(Naming.tensorboard_name('kernel_prediction_' + prediction_feature.name)):
            self.kernel_predictor.predict(prediction_feature)
          
          self.multiscale_predictor.predict(prediction_feature, multiscale_combine_reuse)
          multiscale_combine_reuse = True
          
          # Convert back to the source data format if needed.
          with tf.name_scope('revert_data_format_conversion'):
            self.data_format_reverter.predict(prediction_feature)
    
    # Create the prediction dictionaries to be returned
    target_prediction_feature = None
    for prediction_feature in self.prediction_features:
      if prediction_feature.is_target:
        target_prediction_feature = prediction_feature
        break
    
    prediction_dictionaries = []
    for scale_index in range(len(target_prediction_feature.predictions)):
      prediction_dictionary = {}
      for prediction_feature in self.prediction_features:
        if prediction_feature.is_target:
          prediction_feature.add_prediction_to_dictionary(scale_index, prediction_dictionary)
      prediction_dictionaries.append(prediction_dictionary)
      
    return prediction_dictionaries
