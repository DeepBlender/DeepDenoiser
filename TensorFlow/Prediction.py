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
  
  # The names have to be sorted, otherwise the channels would be randomly mixed.
  feature_names = sorted(list(features.keys()))
  
  prediction_features = []
  for feature_name in feature_names:
    feature = features[feature_name]
    
    # REMARK: It is assumed that there are no features which are only a target, without also being a source.
    if feature['is_source']:
      feature_standardization = feature['standardization']
      feature_standardization = FeatureStandardization(
          feature_standardization['use_log1p'], feature_standardization['mean'], feature_standardization['variance'], feature_name)
      prediction_feature = PredictionFeature(
          feature['is_target'], feature_standardization, feature['number_of_channels'], feature_name)
      prediction_features.append(prediction_feature)
  
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
        
        features[RenderPasses.source_feature_name(prediction_feature.name)] = image
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
          'use_CPU_only': use_CPU_only,
          'data_format': parsed_arguments.data_format})
  
  
  predictions = estimator.predict(input_fn=lambda: input_fn_predict(features, height, width))
        
  for prediction in predictions:
  
    diffuse_direct = prediction[RenderPasses.prediction_feature_name(RenderPasses.DIFFUSE_DIRECT)]
    diffuse_indirect = prediction[RenderPasses.prediction_feature_name(RenderPasses.DIFFUSE_INDIRECT)]
    diffuse_color = prediction[RenderPasses.prediction_feature_name(RenderPasses.DIFFUSE_COLOR)]
    
    glossy_direct = prediction[RenderPasses.prediction_feature_name(RenderPasses.GLOSSY_DIRECT)]
    glossy_indirect = prediction[RenderPasses.prediction_feature_name(RenderPasses.GLOSSY_INDIRECT)]
    glossy_color = prediction[RenderPasses.prediction_feature_name(RenderPasses.GLOSSY_COLOR)]
    
    subsurface_direct = prediction[RenderPasses.prediction_feature_name(RenderPasses.SUBSURFACE_DIRECT)]
    subsurface_indirect = prediction[RenderPasses.prediction_feature_name(RenderPasses.SUBSURFACE_INDIRECT)]
    subsurface_color = prediction[RenderPasses.prediction_feature_name(RenderPasses.SUBSURFACE_COLOR)]
    
    transmission_direct = prediction[RenderPasses.prediction_feature_name(RenderPasses.TRANSMISSION_DIRECT)]
    transmission_indirect = prediction[RenderPasses.prediction_feature_name(RenderPasses.TRANSMISSION_INDIRECT)]
    transmission_color = prediction[RenderPasses.prediction_feature_name(RenderPasses.TRANSMISSION_COLOR)]
    
    environment = prediction[RenderPasses.prediction_feature_name(RenderPasses.ENVIRONMENT)]
    emission = prediction[RenderPasses.prediction_feature_name(RenderPasses.EMISSION)]

  
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