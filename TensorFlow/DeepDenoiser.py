from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import tensorflow as tf
import multiprocessing

import Utilities

import RefinementNet
from UNet import UNet

from LossDifference import LossDifference
from LossDifference import LossDifferenceEnum
from RenderPasses import RenderPasses
import Conv2dUtilities

parser = argparse.ArgumentParser(description='Training and inference for the DeepDenoiser.')

parser.add_argument(
    'json_filename',
    help='The json specifying all the relevant details.')

parser.add_argument(
    '--batch_size', type=int, default=48,
    help='Number of tiles to process in a batch')

parser.add_argument(
    '--threads', default=multiprocessing.cpu_count() + 1,
    help='Number of threads to use')

parser.add_argument(
    '--train_epochs', type=int, default=10000,
    help='Number of epochs to train.')

parser.add_argument(
    '--validation_interval', type=int, default=1,
    help='Number of epochs after which a validation is made.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

def global_activation_function(features, name=None):
  # HACK: Quick way to experiment with other activation function.
  return tf.nn.relu(features, name)
  # return tf.nn.crelu(features, name)
  # return tf.nn.elu(features, name)
  # return tf.nn.selu(features, name)

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

  def standardize(self, feature):
    with tf.name_scope('standardize_' + RenderPasses.tensorboard_name(self.name)):
      if self.use_log1p:
        feature = Utilities.signed_log1p(feature)
      if self.use_mean():
        feature = tf.subtract(feature, self.mean)
      if self.use_variance():
        feature = tf.divide(feature, tf.sqrt(self.variance))
    return feature
    
  def invert_standardize(self, feature):
    with tf.name_scope('invert_standardize_' + RenderPasses.tensorboard_name(self.name)):
      if self.use_variance():
        feature = tf.multiply(feature, tf.sqrt(self.variance))
      if self.use_mean():
        feature = tf.add(feature, self.mean)
      if self.use_log1p:
        feature = Utilities.signed_expm1(feature)
    return feature


class PredictionFeature:

  def __init__(self, is_target, feature_standardization, number_of_channels, name):
    self.is_target = is_target
    self.feature_standardization = feature_standardization
    self.number_of_channels = number_of_channels
    self.name = name

  def initialize_source_from_dictionary(self, dictionary):
    self.source = dictionary[RenderPasses.source_feature_name(self.name)]

  def standardize(self):
    if self.feature_standardization != None:
      self.source = self.feature_standardization.standardize(self.source)

  def prediction_invert_standardize(self):
    if self.feature_standardization != None:
      self.prediction = self.feature_standardization.invert_standardize(self.prediction)
  
  def add_prediction(self, prediction):
    if not self.is_target:
      raise Exception('Adding a prediction for a feature that is not a target is not allowed.')
    self.prediction = prediction
  
  def add_prediction_to_dictionary(self, dictionary):
    if self.is_target:
      dictionary[RenderPasses.prediction_feature_name(self.name)] = self.prediction


class BaseTrainingFeature:

  def __init__(
      self, name, loss_difference, use_difference_of_log1p,
      mean_weight, variation_weight,
      track_mean, track_variation,
      track_difference_histogram, track_variation_difference_histogram):
      
    self.name = name
    self.loss_difference = loss_difference
    self.use_difference_of_log1p = use_difference_of_log1p
    
    self.mean_weight = mean_weight
    self.variation_weight = variation_weight
    
    self.track_mean = track_mean
    self.track_variation = track_variation
    self.track_difference_histogram = track_difference_histogram
    self.track_variation_difference_histogram = track_variation_difference_histogram
  
  
  def difference(self):
    with tf.name_scope('difference'):
      result = LossDifference.difference(self.predicted, self.target, self.loss_difference, use_difference_of_log1p=self.use_difference_of_log1p)
    return result
  
  def horizontal_variation_difference(self):
    with tf.name_scope('horizontal_variation_difference'):
      predicted_horizontal_variation = BaseTrainingFeature.__horizontal_variation(self.predicted)
      target_horizontal_variation = BaseTrainingFeature.__horizontal_variation(self.target)
      result = LossDifference.difference(predicted_horizontal_variation, target_horizontal_variation, self.loss_difference)
    return result
  
  def vertical_variation_difference(self):
    with tf.name_scope('vertical_variation_difference'):
      predicted_vertical_variation = BaseTrainingFeature.__vertical_variation(self.predicted)
      target_vertical_variation = BaseTrainingFeature.__vertical_variation(self.target)
      result = LossDifference.difference(predicted_vertical_variation, target_vertical_variation, self.loss_difference)
    return result
  
  def variation_difference(self):
    with tf.name_scope('variation_difference'):
      result = tf.concat(
          [tf.layers.flatten(self.horizontal_variation_difference()),
          tf.layers.flatten(self.vertical_variation_difference())], axis=1)
    return result
  
  def mean(self):
    if RenderPasses.is_direct_or_indirect_render_pass(self.name):
      result = tf.cond(
          tf.greater(self.mask_sum, 0.),
          lambda: tf.reduce_sum(tf.divide(tf.multiply(self.difference(), self.mask), self.mask_sum)),
          lambda: tf.constant(0.))
    else:
      result = tf.reduce_mean(self.difference())
    return result
  
  def variation(self):
    if RenderPasses.is_direct_or_indirect_render_pass(self.name):
      result = tf.cond(
          tf.greater(self.mask_sum, 0.),
          lambda: tf.reduce_sum(tf.divide(tf.multiply(self.variation_difference(), self.mask), self.mask_sum)),
          lambda: tf.constant(0.))
    else:
      result = tf.reduce_mean(self.variation_difference())
    return result
  
  
  def loss(self):
    with tf.name_scope('loss_' + RenderPasses.tensorboard_name(self.name)):
      result = 0.0
      if self.mean_weight > 0.0:
        with tf.name_scope('loss_' + RenderPasses.mean_name(self.name)):
          result = tf.add(result, tf.scalar_mul(self.mean_weight, self.mean()))
      if self.variation_weight > 0.0:
        with tf.name_scope('loss_' + RenderPasses.variation_name(self.name)):
          result = tf.add(result, tf.scalar_mul(self.variation_weight, self.variation()))
    return result
    
  
  def add_tracked_summaries(self):
    if self.track_mean:
      tf.summary.scalar(RenderPasses.mean_name(self.name), self.mean())
    if self.track_variation:
      tf.summary.scalar(RenderPasses.variation_name(self.name), self.variation())
  
  def add_tracked_histograms(self):
    if self.track_difference_histogram:
      tf.summary.histogram(RenderPasses.tensorboard_name(self.name), self.difference())
    if self.track_variation_difference_histogram:
      tf.summary.histogram(RenderPasses.variation_name(self.name), self.variation_difference())
    
  def add_tracked_metrics_to_dictionary(self, dictionary):
    if self.track_mean:
      dictionary[RenderPasses.mean_name(self.name)] = tf.metrics.mean(self.mean())
    if self.track_variation:
      dictionary[RenderPasses.variation_name(self.name)] = tf.metrics.mean(self.variation())

  @staticmethod
  def __horizontal_variation(image_batch):
    # 'channels_last' or NHWC
    image_batch = tf.subtract(BaseTrainingFeature.__shift_left(image_batch), BaseTrainingFeature.__shift_right(image_batch))
    return image_batch
    
  def __vertical_variation(image_batch):
    # 'channels_last' or NHWC
    image_batch = tf.subtract(BaseTrainingFeature.__shift_up(image_batch), BaseTrainingFeature.__shift_down(image_batch))
    return image_batch
    
  @staticmethod
  def __shift_left(image_batch):
    # 'channels_last' or NHWC
    axis = 2
    width = tf.shape(image_batch)[axis]
    image_batch = tf.slice(image_batch, [0, 0, 1, 0], [-1, -1, width - 1, -1])
    return(image_batch)
  
  @staticmethod
  def __shift_right(image_batch):
    # 'channels_last' or NHWC
    axis = 2
    width = tf.shape(image_batch)[axis]
    image_batch = tf.slice(image_batch, [0, 0, 0, 0], [-1, -1, width - 1, -1]) 
    return(image_batch)
  
  @staticmethod
  def __shift_up(image_batch):
    # 'channels_last' or NHWC
    axis = 1
    height = tf.shape(image_batch)[axis]
    image_batch = tf.slice(image_batch, [0, 1, 0, 0], [-1, height - 1, -1, -1]) 
    return(image_batch)

  @staticmethod
  def __shift_down(image_batch):
    # 'channels_last' or NHWC
    axis = 1
    height = tf.shape(image_batch)[axis]
    image_batch = tf.slice(image_batch, [0, 0, 0, 0], [-1, height - 1, -1, -1]) 
    return(image_batch)


class TrainingFeature(BaseTrainingFeature):

  def __init__(
      self, name, loss_difference, use_difference_of_log1p,
      mean_weight, variation_weight,
      track_mean, track_variation,
      track_difference_histogram, track_variation_difference_histogram):
    
    BaseTrainingFeature.__init__(
        self, name, loss_difference, use_difference_of_log1p,
        mean_weight, variation_weight,
        track_mean, track_variation,
        track_difference_histogram, track_variation_difference_histogram)
  
  def initialize(self, source_features, predicted_features, target_features):
    self.predicted = predicted_features[RenderPasses.prediction_feature_name(self.name)]
    self.target = target_features[RenderPasses.target_feature_name(self.name)]
    if RenderPasses.is_direct_or_indirect_render_pass(self.name):
      corresponding_color_pass = RenderPasses.direct_or_indirect_to_color_render_pass(self.name)
      corresponding_target_feature = target_features[RenderPasses.target_feature_name(corresponding_color_pass)]
      self.mask = Conv2dUtilities.non_zero_mask(corresponding_target_feature, data_format='channels_last')
      self.mask_sum = tf.reduce_sum(self.mask)


class CombinedTrainingFeature(BaseTrainingFeature):

  def __init__(
      self, name, loss_difference, use_difference_of_log1p,
      color_training_feature, direct_training_feature, indirect_training_feature,
      mean_weight, variation_weight,
      track_mean, track_variation,
      track_difference_histogram, track_variation_difference_histogram):
    
    BaseTrainingFeature.__init__(
        self, name, loss_difference, use_difference_of_log1p,
        mean_weight, variation_weight,
        track_mean, track_variation,
        track_difference_histogram, track_variation_difference_histogram)
    
    self.color_training_feature = color_training_feature
    self.direct_training_feature = direct_training_feature
    self.indirect_training_feature = indirect_training_feature
  
  def initialize(self, source_features, predicted_features, target_features):
    self.predicted = tf.multiply(
        self.color_training_feature.predicted,
        tf.add(
            self.direct_training_feature.predicted,
            self.indirect_training_feature.predicted))

    self.target = tf.multiply(
        self.color_training_feature.target,
        tf.add(
            self.direct_training_feature.target,
            self.indirect_training_feature.target))
  
  
class CombinedImageTrainingFeature(BaseTrainingFeature):

  def __init__(
      self, name, loss_difference, use_difference_of_log1p,
      diffuse_training_feature, glossy_training_feature,
      subsurface_training_feature, transmission_training_feature,
      emission_training_feature, environment_training_feature,
      mean_weight, variation_weight,
      track_mean, track_variation,
      track_difference_histogram, track_variation_difference_histogram):
    
    BaseTrainingFeature.__init__(
        self, name, loss_difference, use_difference_of_log1p,
        mean_weight, variation_weight,
        track_mean, track_variation,
        track_difference_histogram, track_variation_difference_histogram)
    
    self.diffuse_training_feature = diffuse_training_feature
    self.glossy_training_feature = glossy_training_feature
    self.subsurface_training_feature = subsurface_training_feature
    self.transmission_training_feature = transmission_training_feature
    self.emission_training_feature = emission_training_feature
    self.environment_training_feature = environment_training_feature
  
  def initialize(self, source_features, predicted_features, target_features):
    self.predicted = tf.add_n([
        self.diffuse_training_feature.predicted,
        self.glossy_training_feature.predicted,
        self.subsurface_training_feature.predicted,
        self.transmission_training_feature.predicted,
        self.emission_training_feature.predicted,
        self.environment_training_feature.predicted])

    self.target = tf.add_n([
        self.diffuse_training_feature.target,
        self.glossy_training_feature.target,
        self.subsurface_training_feature.target,
        self.transmission_training_feature.target,
        self.emission_training_feature.target,
        self.environment_training_feature.target])

class TrainingFeaturePreparation:

  def __init__(self, is_target, number_of_channels, name):
    self.is_target = is_target
    self.number_of_channels = number_of_channels
    self.name = name
  
  def add_to_parse_dictionary(self, dictionary, index):
    dictionary[RenderPasses.source_feature_name_indexed(self.name, index)] = tf.FixedLenFeature([], tf.string)
    if self.is_target:
      dictionary[RenderPasses.target_feature_name(self.name)] = tf.FixedLenFeature([], tf.string)
  
  def deserialize(self, parsed_features, height, width, index):
    self.source = tf.decode_raw(parsed_features[RenderPasses.source_feature_name_indexed(self.name, index)], tf.float32)
    self.source = tf.reshape(self.source, [height, width, self.number_of_channels])
    if self.is_target:
      self.target = tf.decode_raw(parsed_features[RenderPasses.target_feature_name(self.name)], tf.float32)
      self.target = tf.reshape(self.target, [height, width, self.number_of_channels])
  
  def flip_left_right(self):
    self.source = tf.image.flip_left_right(self.source)
    if self.is_target:
      self.target = tf.image.flip_left_right(self.target)
  
  def rotate90(self, k):
    self.source = tf.image.rot90(self.source, k=k)
    if self.is_target:
      self.target = tf.image.rot90(self.target, k=k)
  
  def add_to_features_dictionary(self, features):
    features[RenderPasses.source_feature_name(self.name)] = self.source
    
  def add_to_targets_dictionary(self, targets):
    if self.is_target:
      targets[RenderPasses.target_feature_name(self.name)] = self.target


def model(prediction_features, mode, use_CPU_only, data_format):
  
  # Standardization of the data
  with tf.name_scope('standardize'):
    for prediction_feature in prediction_features:
      prediction_feature.standardize()

  with tf.name_scope('concat_all_features'):
    concat_axis = 3
    prediction_inputs = []
    auxiliary_inputs = []
    for prediction_feature in prediction_features:
      if prediction_feature.is_target:
        prediction_inputs.append(prediction_feature.source)
      else:
        auxiliary_inputs.append(prediction_feature.source)
    
    prediction_inputs = tf.concat(prediction_inputs, concat_axis)
    auxiliary_inputs = tf.concat(auxiliary_inputs, concat_axis)
  
  
  is_training = False
  if mode == tf.estimator.ModeKeys.TRAIN:
    is_training = True
  
  if data_format is None:
    # When running on GPU, transpose the data from channels_last (NHWC) to
    # channels_first (NCHW) to improve performance.
    # See https://www.tensorflow.org/performance/performance_guide#data_formats
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else
        'channels_last')
    if use_CPU_only:
      data_format = 'channels_last'

  concat_axis = 3
  if data_format == 'channels_first':
    prediction_inputs = tf.transpose(prediction_inputs, [0, 3, 1, 2])
    auxiliary_inputs = tf.transpose(auxiliary_inputs, [0, 3, 1, 2])
    concat_axis = 1
  
  
  reshape_output = True
  invert_standardize = False
  
  with tf.name_scope('model'):
  
    unet = UNet(number_of_initial_convolution_channels=256, number_of_sampling_steps=0, number_of_convolutions_per_block=2)
    outputs = unet.u_net(prediction_inputs, auxiliary_inputs, data_format=data_format)
    reshape_output = True
    invert_standardize = False
  
    # _refinement_net = RefinementNet.RefinementNet(
        # number_of_repetitions=0, number_of_blocks=1, number_of_convolutions_per_block=8, number_block_repetitions=0, number_of_temporary_data_filters=0, number_of_filters_per_convolution=32, activation_function=global_activation_function, use_zero_padding=True, use_channel_weighting=False)
    # outputs = _refinement_net.refinement_net(prediction_inputs, auxiliary_inputs, is_training, data_format=data_format)
    # reshape_output = False
    # invert_standardize = True
  
  output_size = 0
  output_prediction_features = []
  for prediction_feature in prediction_features:
    if prediction_feature.is_target:
      output_size = output_size + prediction_feature.number_of_channels
      output_prediction_features.append(prediction_feature)
  
  if reshape_output:
    reshape_kernel_size = 3
    # Reshape to get the correct number of channels.
    outputs = Conv2dUtilities.convolution2d(
        inputs=outputs,
        filters=output_size,
        kernel_size=[reshape_kernel_size, reshape_kernel_size],
        activation=global_activation_function,
        data_format=data_format, name='reshape')
  
  if data_format == 'channels_first':
    outputs = tf.transpose(outputs, [0, 2, 3, 1])
  
  
  concat_axis = 3
  size_splits = []
  for prediction_feature in output_prediction_features:
    size_splits.append(prediction_feature.number_of_channels)
  
  with tf.name_scope('split'):
    prediction_tuple = tf.split(outputs, size_splits, concat_axis)
  for index, prediction in enumerate(prediction_tuple):
    output_prediction_features[index].add_prediction(prediction)
  
  if invert_standardize:
    for prediction_feature in output_prediction_features:
      prediction_feature.prediction_invert_standardize()
  
  prediction_dictionary = {}
  for prediction_feature in output_prediction_features:
    prediction_feature.add_prediction_to_dictionary(prediction_dictionary)
  
  return prediction_dictionary

def model_fn(features, labels, mode, params):
  prediction_features = params['prediction_features']
  
  for prediction_feature in prediction_features:
    prediction_feature.initialize_source_from_dictionary(features)
  
  data_format = params['data_format']
  predictions = model(prediction_features, mode, params['use_CPU_only'], data_format)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  targets = labels
  
  with tf.name_scope('loss_function'):
    with tf.name_scope('feature_loss'):
      training_features = params['training_features']
      for training_feature in training_features:
        training_feature.initialize(features, predictions, targets)
      feature_losses = []
      for training_feature in training_features:
        feature_losses.append(training_feature.loss())
      if len(feature_losses) > 0:
        feature_loss = tf.add_n(feature_losses)
      else:
        feature_loss = 0.0
    
    with tf.name_scope('combined_feature_loss'):
      combined_training_features = params['combined_training_features']
      if combined_training_features != None:
        for combined_training_feature in combined_training_features:
          combined_training_feature.initialize(features, predictions, targets)
        combined_feature_losses = []
        for combined_training_feature in combined_training_features:
          combined_feature_losses.append(combined_training_feature.loss())
        if len(combined_feature_losses) > 0:
          combined_feature_loss = tf.add_n(combined_feature_losses)
      else:
        combined_feature_loss = 0.0
    
    with tf.name_scope('combined_image_loss'):
      combined_image_training_feature = params['combined_image_training_feature']
      if combined_image_training_feature != None:
        combined_image_training_feature.initialize(features, predictions, targets)
        combined_image_feature_loss = combined_image_training_feature.loss()
      else:
        combined_image_feature_loss = 0.0
    
    # All losses combined
    loss = tf.add_n([feature_loss, combined_feature_loss, combined_image_feature_loss])

  
  # Configure the training op
  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = params['learning_rate']
    global_step = tf.train.get_or_create_global_step()
    first_decay_steps = 1000
    t_mul = 1.3 # Use t_mul more steps after each restart.
    m_mul = 0.8 # Multiply the learning rate after each restart with this number.
    alpha = 1 / 1000. # Learning rate decays from 1 * learning_rate to alpha * learning_rate.
    learning_rate_decayed = tf.train.cosine_decay_restarts(learning_rate, global_step, first_decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha)
  
    tf.summary.scalar('learning_rate', learning_rate_decayed)
    tf.summary.scalar('batch_size', params['batch_size'])
    
    # Histograms
    for training_feature in training_features:
      training_feature.add_tracked_histograms()
    if combined_training_features != None:
      for combined_training_feature in combined_training_features:
        combined_training_feature.add_tracked_histograms()
    if combined_image_training_feature != None:
      combined_image_training_feature.add_tracked_histograms()
    
    # Summaries
    #with tf.name_scope('features'):
    for training_feature in training_features:
      training_feature.add_tracked_summaries()
    #with tf.name_scope('combined_features'):
    if combined_training_features != None:
      for combined_training_feature in combined_training_features:
        combined_training_feature.add_tracked_summaries()
    #with tf.name_scope('combined'):
    if combined_image_training_feature != None:
      combined_image_training_feature.add_tracked_summaries()
    
    optimizer = tf.train.AdamOptimizer(learning_rate_decayed)
    train_op = optimizer.minimize(loss, global_step)
    eval_metric_ops = None
  else:
    train_op = None
    eval_metric_ops = {}

    #with tf.name_scope('features'):
    for training_feature in training_features:
      training_feature.add_tracked_metrics_to_dictionary(eval_metric_ops)
    
    #with tf.name_scope('combined_features'):
    if combined_training_features != None:
      for combined_training_feature in combined_training_features:
        combined_training_feature.add_tracked_metrics_to_dictionary(eval_metric_ops)
    
    #with tf.name_scope('combined'):
    if combined_image_training_feature != None:
      combined_image_training_feature.add_tracked_metrics_to_dictionary(eval_metric_ops)
    
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def input_fn_tfrecords(files, training_features_preparation, number_of_epochs, number_of_sources_per_example, tiles_height_width, batch_size, threads, use_data_augmentation=True):
  
  def feature_parser(serialized_example):
  
    # TODO: Split it up (DeepBlender)
    # - Prepare, parse and deserialize them and create the datasets within a flat_map
    # - Add and augmentation pass which uses map and is multithreaded
  
    dataset = None
    
    for index in range(number_of_sources_per_example):
      
      # skip = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]
      # if skip == 0:
        # continue
      
      features = {}

      for training_feature_preparation in training_features_preparation:
        training_feature_preparation.add_to_parse_dictionary(features, index)
    
      parsed_features = tf.parse_single_example(serialized_example, features)
      
      for training_feature_preparation in training_features_preparation:
        training_feature_preparation.deserialize(parsed_features, tiles_height_width, tiles_height_width, index)
      
      
      # Data augmentation
      
      if use_data_augmentation:
        with tf.name_scope('data_augmentation'):
          
          # Flip the image randomly (REMARK: maxval is excluded in random_uniform!)
          flip = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]
          if flip != 0:
            for training_feature_preparation in training_features_preparation:
              training_feature_preparation.flip_left_right()
              if training_feature_preparation.name == RenderPasses.SCREEN_SPACE_NORMAL:
                screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(training_feature_preparation.source, [1, 1, 1], 2)
                screen_space_normal_x = tf.negative(screen_space_normal_x)
                training_feature_preparation.source = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], 2)
                

          # Rotate the image randomly (maxval is excluded!)
          rotate = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)[0]
          if rotate != 0:
            for training_feature_preparation in training_features_preparation:
              training_feature_preparation.rotate90(rotate)
              if training_feature_preparation.name == RenderPasses.SCREEN_SPACE_NORMAL:
                screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(training_feature_preparation.source, [1, 1, 1], 2)
                if rotate == 1:
                  # x -> -y
                  # y -> x
                  temporary_screen_space_normal_x = screen_space_normal_x
                  screen_space_normal_x = tf.negative(screen_space_normal_y)
                  screen_space_normal_y = temporary_screen_space_normal_x
                elif rotate == 2:
                  # x -> -x
                  # y -> -y
                  screen_space_normal_x = tf.negative(screen_space_normal_x)
                  screen_space_normal_y = tf.negative(screen_space_normal_y)
                elif rotate == 3:
                  # x -> y
                  # y -> -x
                  temporary_screen_space_normal_y = screen_space_normal_y
                  screen_space_normal_y = tf.negative(screen_space_normal_x)
                  screen_space_normal_x = temporary_screen_space_normal_y
                  
                training_feature_preparation.source = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], 2)
        
      features = {}
      for training_feature_preparation in training_features_preparation:
        training_feature_preparation.add_to_features_dictionary(features)
      
      targets = {}
      for training_feature_preparation in training_features_preparation:
        training_feature_preparation.add_to_targets_dictionary(targets)
      
      if dataset == None:
        dataset = tf.data.Dataset.from_tensors((features, targets))
      else:
        dataset = dataset.concatenate(tf.data.Dataset.from_tensors((features, targets)))
    
    return dataset
  
  
  # REMARK: Due to stability issues, it was not possible to follow all the suggestions from the documentation like using the fused versions.
  
  shuffle_buffer_size = 10000
  files = files.repeat(number_of_epochs)
  files = files.shuffle(buffer_size=shuffle_buffer_size)
  
  dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', buffer_size=None, num_parallel_reads=threads)
  dataset = dataset.flat_map(map_func=feature_parser)#, num_parallel_calls=threads)
  
  shuffle_buffer_size = 40 * batch_size
  dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  
  dataset = dataset.batch(batch_size)
  
  prefetch_buffer_size = 1
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
  
  iterator = dataset.make_one_shot_iterator()
  
  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `target` is a batch of targets.
  features, targets = iterator.get_next()
  return features, targets


def train(tfrecords_directory, estimator, training_features_preparation, number_of_epochs, number_of_sources_per_example, tiles_height_width, batch_size, threads):
  files = tf.data.Dataset.list_files(tfrecords_directory + '/*')

  # Train the model
  estimator.train(input_fn=lambda: input_fn_tfrecords(files, training_features_preparation, number_of_epochs, number_of_sources_per_example, tiles_height_width, batch_size, threads))

def evaluate(tfrecords_directory, estimator, training_features_preparation, number_of_sources_per_example, tiles_height_width, batch_size, threads):
  files = tf.data.Dataset.list_files(tfrecords_directory + '/*')

  # Evaluate the model
  estimator.evaluate(input_fn=lambda: input_fn_tfrecords(files, training_features_preparation, 1, number_of_sources_per_example, tiles_height_width, batch_size, threads))

def main(parsed_arguments):
  if not isinstance(parsed_arguments.threads, int):
    parsed_arguments.threads = int(parsed_arguments.threads)

  try:
    json_filename = parsed_arguments.json_filename
    json_content = open(json_filename, 'r').read()
    parsed_json = json.loads(json_content)
  except:
    print('Expected a valid json file as argument.')
  
  model_directory = parsed_json['model_directory']
  base_tfrecords_directory = parsed_json['base_tfrecords_directory']
  modes = parsed_json['modes']
  
  loss_difference = parsed_json['loss_difference']
  loss_difference = LossDifferenceEnum[loss_difference]
  use_difference_of_log1p = parsed_json['use_difference_of_log1p']
  
  features = parsed_json['features']
  combined_features = parsed_json['combined_features']
  combined_image = parsed_json['combined_image']
  
  # The names have to be sorted, otherwise the channels would be randomly mixed.
  feature_names = sorted(list(features.keys()))
  
  prediction_features = []
  for feature_name in feature_names:
    feature = features[feature_name]
    
    # REMARK: It is assumed that there are no features which are only a target, without also being a source.
    if feature['is_source']:
      feature_standardization = feature['standardization']
      feature_standardization = FeatureStandardization(feature_standardization['use_log1p'], feature_standardization['mean'], feature_standardization['variance'], feature_name)
      prediction_feature = PredictionFeature(feature['is_target'], feature_standardization, feature['number_of_channels'], feature_name)
      prediction_features.append(prediction_feature)
  
  
  # Training features.

  training_features = []
  feature_name_to_training_feature = {}
  for feature_name in feature_names:
    feature = features[feature_name]
    if feature['is_source'] and feature['is_target']:
      statistics = feature['statistics']
      loss_weights = feature['loss_weights']
      training_feature = TrainingFeature(
          feature_name, loss_difference, use_difference_of_log1p,
          loss_weights['mean'], loss_weights['variation'],
          statistics['track_mean'], statistics['track_variation'],
          statistics['track_difference_histogram'], statistics['track_variation_difference_histogram'])
      training_features.append(training_feature)
      feature_name_to_training_feature[feature_name] = training_feature

  training_features_preparation = []
  for prediction_feature in prediction_features:
    training_features_preparation.append(TrainingFeaturePreparation(prediction_feature.is_target, prediction_feature.number_of_channels, prediction_feature.name))

  
  # Combined training features.
  
  combined_training_features = []
  combined_feature_name_to_combined_training_feature = {}
  combined_feature_names = list(combined_features.keys())
  for combined_feature_name in combined_feature_names:
    combined_feature = combined_features[combined_feature_name]
    statistics = combined_feature['statistics']
    loss_weights = combined_feature['loss_weights']
    if loss_weights['mean'] > 0. or loss_weights['variation'] > 0:
      color_feature_name = RenderPasses.combined_to_color_render_pass(combined_feature_name)
      direct_feature_name = RenderPasses.combined_to_direct_render_pass(combined_feature_name)
      indirect_feature_name = RenderPasses.combined_to_indirect_render_pass(combined_feature_name)
      combined_training_feature = CombinedTrainingFeature(
          combined_feature_name, loss_difference, use_difference_of_log1p,
          feature_name_to_training_feature[color_feature_name],
          feature_name_to_training_feature[direct_feature_name],
          feature_name_to_training_feature[indirect_feature_name],
          loss_weights['mean'], loss_weights['variation'],
          statistics['track_mean'], statistics['track_variation'],
          statistics['track_difference_histogram'], statistics['track_variation_difference_histogram'])
      combined_training_features.append(combined_training_feature)
      combined_feature_name_to_combined_training_feature[combined_feature_name] = combined_training_feature
        
  if len(combined_training_features) == 0:
    combined_training_features = None
  
  
  # Combined image training feature.
  
  combined_image_training_feature = None
  statistics = combined_image['statistics']
  loss_weights = combined_image['loss_weights']
  if loss_weights['mean'] > 0. or loss_weights['variation'] > 0:
    combined_image_training_feature = CombinedImageTrainingFeature(
        RenderPasses.COMBINED, loss_difference, use_difference_of_log1p,
        combined_feature_name_to_combined_training_feature['Diffuse'],
        combined_feature_name_to_combined_training_feature['Glossy'],
        combined_feature_name_to_combined_training_feature['Subsurface'],
        combined_feature_name_to_combined_training_feature['Transmission'],
        feature_name_to_training_feature[RenderPasses.EMISSION],
        feature_name_to_training_feature[RenderPasses.ENVIRONMENT],
        loss_weights['mean'], loss_weights['variation'],
        statistics['track_mean'], statistics['track_variation'],
        statistics['track_difference_histogram'], statistics['track_variation_difference_histogram'])
  
  
  # TODO: CPU only has to be configurable.
  # TODO: Learning rate has to be configurable.
  
  learning_rate = 1e-4
  use_XLA = True
  use_CPU_only = False
  
  run_config = None
  if use_XLA:
    if use_CPU_only:
      session_config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
      session_config = tf.ConfigProto()
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    save_summary_steps = 100
    save_checkpoints_step = 500
    run_config = tf.estimator.RunConfig(session_config=session_config, save_summary_steps=save_summary_steps, save_checkpoints_steps=save_checkpoints_step)
  
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=model_directory,
      config=run_config,
      params={
          'prediction_features': prediction_features,
          'use_CPU_only': use_CPU_only,
          'data_format': parsed_arguments.data_format,
          'learning_rate': learning_rate,
          'batch_size': parsed_arguments.batch_size,
          'training_features': training_features,
          'combined_training_features': combined_training_features,
          'combined_image_training_feature': combined_image_training_feature})
  
  training_tfrecords_directory = os.path.join(base_tfrecords_directory, 'training')
  validation_tfrecords_directory = os.path.join(base_tfrecords_directory, 'validation')
  
  if not 'training' in modes:
    raise Exception('No training mode found.')
  if not 'validation' in modes:
    raise Exception('No validation mode found.')
  training_statistics_filename = os.path.join(base_tfrecords_directory, 'training.json')
  validation_statistics_filename = os.path.join(base_tfrecords_directory, 'validation.json')
  
  training_statistics_content = open(training_statistics_filename, 'r').read()
  training_statistics = json.loads(training_statistics_content)
  validation_statistics_content = open(validation_statistics_filename, 'r').read()
  validation_statistics = json.loads(validation_statistics_content)
  
  training_tiles_height_width = training_statistics['tiles_height_width']
  training_number_of_sources_per_example = training_statistics['number_of_sources_per_example']
  validation_tiles_height_width = validation_statistics['tiles_height_width']
  validation_number_of_sources_per_example = validation_statistics['number_of_sources_per_example']
  
  remaining_epochs = parsed_arguments.train_epochs
  while remaining_epochs > 0:
    current_epochs = parsed_arguments.validation_interval
    if remaining_epochs < current_epochs:
      current_epochs = remaining_epochs
    
    train(training_tfrecords_directory, estimator, training_features_preparation, current_epochs, training_number_of_sources_per_example, training_tiles_height_width, parsed_arguments.batch_size, parsed_arguments.threads)
    evaluate(validation_tfrecords_directory, estimator, training_features_preparation, validation_number_of_sources_per_example, training_tiles_height_width, parsed_arguments.batch_size, parsed_arguments.threads)
    
    remaining_epochs = remaining_epochs - current_epochs


if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)