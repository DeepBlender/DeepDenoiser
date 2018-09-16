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

from Architecture import Architecture

from RenderPasses import RenderPasses
from Naming import Naming
from OpenEXRDirectory import OpenEXRDirectory

parser = argparse.ArgumentParser(description='Prediction for the DeepDenoiser.')

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
    '--data_format', type=str, default='channels_first',
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

def model_fn(features, labels, mode, params):
  architecture = params['architecture']
  predictions = architecture.predict(features, mode)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = predictions[0]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def main(parsed_arguments):
  if not isinstance(parsed_arguments.threads, int):
    parsed_arguments.threads = int(parsed_arguments.threads)

  try:
    architecture_json_filename = parsed_arguments.json_filename
    architecture_json_content = open(architecture_json_filename, 'r').read()
    parsed_architecture_json = json.loads(architecture_json_content)
  except:
    print('Expected a valid architecture json file.')
  
  assert os.path.isdir(parsed_arguments.input)
  
  
  architecture = Architecture(parsed_architecture_json, source_data_format='channels_last', data_format=parsed_arguments.data_format)
  if architecture.data_format == 'channels_first':
    use_CPU_only = False
  else:
    use_CPU_only = True
  
  height = None
  width = None
  
  exr_files = OpenEXRDirectory._exr_files(parsed_arguments.input)
  features = {}
  for prediction_feature in architecture.prediction_features:
    exr_loaded = False
    for exr_file in exr_files:
      if prediction_feature.name in exr_file:
        image = OpenEXRDirectory._load_exr(exr_file)
        
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
  

  if use_CPU_only:
    print('cpu')
    session_config = tf.ConfigProto(device_count = {'GPU': 0})
  else:
    session_config = tf.ConfigProto()

  use_XLA = True
  if use_XLA:
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  
  run_config = tf.estimator.RunConfig(session_config=session_config)
  
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=architecture.model_directory,
      config=run_config,
      params={'architecture': architecture})
  
  
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