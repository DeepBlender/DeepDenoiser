from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import cv2
import numpy as np
import tensorflow as tf
import multiprocessing

from DeepDenoiser import *

from RenderPasses import RenderPasses
from Naming import Naming
from RenderDirectory import RenderDirectory

parser = argparse.ArgumentParser(description='Training and inference for the DeepDenoiser.')

parser.add_argument(
    'json_filename',
    help='The json specifying all the relevant details.')

parser.add_argument(
    '--input', type=str,
    help='Make a prediction for the files in this directory.')

parser.add_argument(
    '--threads', default=multiprocessing.cpu_count() + 1,
    help='Number of threads to use')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')


def input_fn_predict(features, height, width):

  # TODO: Create a dataset as needed (DeepBlender)
  
  for feature_name in features:
    image = features[feature_name]
    image = tf.convert_to_tensor(image, np.float32)
    if len(image.shape) == 2:
      image = tf.reshape(image, [-1, height, width, 1])
    else:
      image = tf.reshape(image, [-1, height, width, 3])
    
    features[feature_name] = image
  
  return (features, 0)

def main(parsed_arguments):
  if not isinstance(parsed_arguments.threads, int):
    parsed_arguments.threads = int(parsed_arguments.threads)

  try:
    json_filename = parsed_arguments.json_filename
    json_content = open(json_filename, 'r').read()
    parsed_json = json.loads(json_content)
  except:
    print('Expected a valid json file as argument.')
  
  assert os.path.isdir(parsed_arguments.input)
  
  model_directory = parsed_json['model_directory']
  features = parsed_json['features']
  
  
  neural_network = parsed_json['neural_network']
  architecture = neural_network['architecture']
  number_of_filters_for_convolution_blocks = neural_network['number_of_filters_for_convolution_blocks']
  number_of_convolutions_per_block = neural_network['number_of_convolutions_per_block']
  use_batch_normalization = neural_network['use_batch_normalization']
  dropout_rate = neural_network['dropout_rate']
  number_of_sources_per_target = neural_network['number_of_sources_per_target']
  use_single_feature_prediction = neural_network['use_single_feature_prediction']
  feature_flags = FeatureFlags(neural_network['feature_flags'])
  use_multiscale_predictions = neural_network['use_multiscale_predictions']
  use_kernel_predicion = neural_network['use_kernel_predicion']
  kernel_size = neural_network['kernel_size']
  
  neural_network = NeuralNetwork(
      architecture=architecture, number_of_filters_for_convolution_blocks=number_of_filters_for_convolution_blocks,
      number_of_convolutions_per_block=number_of_convolutions_per_block, use_batch_normalization=use_batch_normalization,
      dropout_rate=dropout_rate, number_of_sources_per_target=number_of_sources_per_target, use_single_feature_prediction=use_single_feature_prediction,
      feature_flags=feature_flags, use_multiscale_predictions=use_multiscale_predictions,
      use_kernel_predicion=use_kernel_predicion, kernel_size=kernel_size)
  
  # The names have to be sorted, otherwise the channels would be randomly mixed.
  feature_names = sorted(list(features.keys()))
  
  prediction_features = []
  for feature_name in feature_names:
    feature = features[feature_name]
    
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
      preserve_source = not invert_standardization
      prediction_feature = PredictionFeature(
          number_of_sources_per_target, preserve_source, feature['is_target'], feature_standardization, invert_standardization, feature_variance,
          feature['feature_flags'], feature['number_of_channels'], feature_name)
      prediction_features.append(prediction_feature)
  
  if use_single_feature_prediction:
    for prediction_feature in prediction_features:
      if prediction_feature.is_target:
        feature_flags.add_render_pass_name_to_feature_flag_names(prediction_feature.name, prediction_feature.feature_flag_names)
    feature_flags.freeze()
  
  height = None
  width = None
  
  exr_files = RenderDirectory._exr_files(parsed_arguments.input)
  features = {}
  for prediction_feature in prediction_features:
    exr_loaded = False
    for exr_file in exr_files:
      if prediction_feature.name in exr_file:
        image = RenderDirectory._load_exr(exr_file)
        
        # Special cases: Alpha and depth passes only have one channel.
        if RenderPasses.number_of_channels(prediction_feature.name) == 1:
          image = image[:, :, 0]
        
        # HACK: Assume just one source input!
        features[Naming.source_feature_name(prediction_feature.name, index=0)] = image
        exr_loaded = True
        
        if height == None:
          height = image.shape[0]
          width = image.shape[1]
        else:
          assert height == image.shape[0]
          assert width == image.shape[1]
        
        break
    if not exr_loaded:
      # TODO: Improve (DeepBlender)
      raise Exception('Image for \'' + render_pass + '\' could not be loaded or does not exist.')
  
  use_XLA = True
  use_CPU_only = True
  
  run_config = None
  if use_XLA:
    if use_CPU_only:
      session_config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
      session_config = tf.ConfigProto()
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    run_config = tf.estimator.RunConfig(session_config=session_config)
  
  
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=model_directory,
      config=run_config,
      params={
          'prediction_features': prediction_features,
          'neural_network': neural_network,
          'use_CPU_only': use_CPU_only,
          'data_format': parsed_arguments.data_format})
  
  
  predictions = estimator.predict(input_fn=lambda: input_fn_predict(features, height, width))
        
  for prediction in predictions:
  
    diffuse_direct = prediction[Naming.prediction_feature_name(RenderPasses.DIFFUSE_DIRECT)]
    diffuse_indirect = prediction[Naming.prediction_feature_name(RenderPasses.DIFFUSE_INDIRECT)]
    diffuse_color = prediction[Naming.prediction_feature_name(RenderPasses.DIFFUSE_COLOR)]
    
    glossy_direct = prediction[Naming.prediction_feature_name(RenderPasses.GLOSSY_DIRECT)]
    glossy_indirect = prediction[Naming.prediction_feature_name(RenderPasses.GLOSSY_INDIRECT)]
    glossy_color = prediction[Naming.prediction_feature_name(RenderPasses.GLOSSY_COLOR)]
    
    subsurface_direct = prediction[Naming.prediction_feature_name(RenderPasses.SUBSURFACE_DIRECT)]
    subsurface_indirect = prediction[Naming.prediction_feature_name(RenderPasses.SUBSURFACE_INDIRECT)]
    subsurface_color = prediction[Naming.prediction_feature_name(RenderPasses.SUBSURFACE_COLOR)]
    
    transmission_direct = prediction[Naming.prediction_feature_name(RenderPasses.TRANSMISSION_DIRECT)]
    transmission_indirect = prediction[Naming.prediction_feature_name(RenderPasses.TRANSMISSION_INDIRECT)]
    transmission_color = prediction[Naming.prediction_feature_name(RenderPasses.TRANSMISSION_COLOR)]
    
    environment = prediction[Naming.prediction_feature_name(RenderPasses.ENVIRONMENT)]
    emission = prediction[Naming.prediction_feature_name(RenderPasses.EMISSION)]

  
    # Combined features
    diffuse = np.multiply(diffuse_color, np.add(diffuse_direct, diffuse_indirect))
    glossy = np.multiply(glossy_color, np.add(glossy_direct, glossy_indirect))
    subsurface = np.multiply(subsurface_color, np.add(subsurface_direct, subsurface_indirect))
    transmission = np.multiply(transmission_color, np.add(transmission_direct, transmission_indirect))
    
    # Combined image
    image = np.add(diffuse, glossy)
    image = np.add(image, subsurface)
    image = np.add(image, transmission)
    image = np.add(image, environment)
    image = np.add(image, emission)
    
    
    # HACK: Temporary output as png. (DeepBlender)
    image = 255. * image
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(parsed_arguments.input + '/output.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    
    # HACK: Break needed because no dataset is used (DeepBlender)
    break


if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)