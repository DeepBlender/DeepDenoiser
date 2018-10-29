from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import cv2
import numpy as np
import math
import tensorflow as tf
import multiprocessing

from Architecture import Architecture
from Architecture import FeaturePredictionType

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
    '--tile_size', default=128,
    help='Width and heights of the tiles into which the image is split before denoising.')

parser.add_argument(
    '--tile_overlap_size', default=14,
    help='Border size of the tiles that is overlapping to avoid artifacts.')


parser.add_argument(
    '--threads', default=multiprocessing.cpu_count() + 1,
    help='Number of threads to use.')

parser.add_argument(
    '--data_format', type=str, default='channels_first',
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')


class FeatureLoader:

  def __init__(self, feature_prediction):
    self.feature_prediction = feature_prediction
  
  def add_to_parse_dictionary(self, dictionary):
    if self.feature_prediction.load_data:
      dictionary[Naming.source_feature_name(self.feature_prediction.name, index=0)] = tf.FixedLenFeature([], tf.string)

  def deserialize(self, parsed_features, height, width):
    if self.feature_prediction.load_data:
      internal_source = tf.decode_raw(
          parsed_features[Naming.source_feature_name(self.feature_prediction.name, index=0)], tf.float32)
      #internal_source = tf.reshape(internal_source, [height, width, self.feature_prediction.number_of_channels])
      internal_source = tf.reshape(internal_source, [height, width, 3])
      self.source = internal_source
  
  def add_to_sources_dictionary(self, sources, height, width):
    if self.feature_prediction.load_data:
      sources[Naming.source_feature_name(self.feature_prediction.name, index=0)] = self.source
    else:
      assert self.feature_prediction.feature_prediction_type != FeaturePredictionType.AUXILIARY
      source = tf.ones([height, width, self.feature_prediction.number_of_channels])
      if self.feature_prediction.feature_prediction_type != FeaturePredictionType.COLOR:
        # Direct and indirect need to be 0.5.
        source = tf.scalar_mul(0.5, source)
      sources[Naming.source_feature_name(self.feature_prediction.name, index=0)] = source


def input_fn_tfrecords(
    files, features_loader, feature_flags,
    tiles_height_width, batch_size, threads, data_format='channels_last'):

  def fast_feature_parser(serialized_example):
    
    # Load all the required indices.
    features = {}
    for feature_loader in features_loader:
      feature_loader.add_to_parse_dictionary(features)
    
    parsed_features = tf.parse_single_example(serialized_example, features)
    
    for feature_loader in features_loader:
      feature_loader.deserialize(parsed_features, tiles_height_width, tiles_height_width)
    
    # Prepare the examples.
    
    sources = {}
    for feature_loader in features_loader:
      feature_loader.add_to_sources_dictionary(sources, tiles_height_width, tiles_height_width)

      if feature_flags != None:
        feature_flags.add_to_source_dictionary(sources, tiles_height_width, tiles_height_width)

    return sources
  
  def feature_parser(serialized_example):
    dataset = None

    # Load all the required indices.
    features = {}
    for feature_loader in features_loader:
      feature_loader.add_to_parse_dictionary(features)

    parsed_features = tf.parse_single_example(serialized_example, features)

    for feature_loader in features_loader:
      feature_loader.deserialize(parsed_features, tiles_height_width, tiles_height_width)

    # Prepare the examples.
    sources = {}
    for feature_loader in features_loader:
      feature_loader.add_to_sources_dictionary(sources, tiles_height_width, tiles_height_width)

      if feature_flags != None:
        feature_flags.add_to_source_dictionary(sources, tiles_height_width, tiles_height_width)

    if dataset == None:
      dataset = tf.data.Dataset.from_tensors((sources))
    else:
      dataset = dataset.concatenate(tf.data.Dataset.from_tensors((sources)))

    return dataset
  
  dataset = tf.data.TFRecordDataset(files, compression_type=None, buffer_size=None, num_parallel_reads=threads)

  dataset = dataset.map(map_func=fast_feature_parser, num_parallel_calls=threads)
  #dataset = dataset.flat_map(map_func=feature_parser)

  dataset = dataset.batch(batch_size)

  prefetch_buffer_size = 5
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
  
  iterator = dataset.make_one_shot_iterator()

  features = iterator.get_next()
  return features


def slow_direct_input_fn_predict(features_list, height, width):
  
  dataset = None
  for features in features_list:
    for feature_name in features:
      image = features[feature_name]
      image = tf.convert_to_tensor(image, np.float32)
      if len(image.shape) == 2:
        image = tf.reshape(image, [-1, height, width, 1])
      else:
        image = tf.reshape(image, [-1, height, width, 3])
      features[feature_name] = image
    current_dataset = tf.data.Dataset.from_tensor_slices(features)
    if dataset == None:
      dataset = current_dataset
    else:
      dataset = dataset.concatenate(current_dataset)

  dataset = dataset.batch(1)
  iterator = dataset.make_one_shot_iterator()
  result = iterator.get_next()
  return result


def model_fn(features, labels, mode, params):
  architecture = params['architecture']
  predictions = architecture.predict(features, mode)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = predictions[0]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def main(parsed_arguments):
  # Eager execution was faster, but the reason was no clear. (DeepBlender)
  tf.enable_eager_execution()

  if not isinstance(parsed_arguments.threads, int):
    parsed_arguments.threads = int(parsed_arguments.threads)

  try:
    architecture_json_filename = parsed_arguments.json_filename
    architecture_json_content = open(architecture_json_filename, 'r').read()
    parsed_architecture_json = json.loads(architecture_json_content)
  except:
    print('Expected a valid architecture json file.')
  
  assert os.path.isdir(parsed_arguments.input)

  if not isinstance(parsed_arguments.tile_size, int):
    parsed_arguments.tile_size = int(parsed_arguments.tile_size)
  if not isinstance(parsed_arguments.tile_overlap_size, int):
    parsed_arguments.tile_overlap_size = int(parsed_arguments.tile_overlap_size)

  tile_size = parsed_arguments.tile_size
  tile_overlap_size = parsed_arguments.tile_overlap_size

  data_format = parsed_arguments.data_format

  architecture = Architecture(parsed_architecture_json, source_data_format='channels_last', data_format=data_format)
  if architecture.data_format == 'channels_first':
    use_CPU_only = False
  else:
    use_CPU_only = True

  height = None
  width = None
  
  exr_files = OpenEXRDirectory._exr_files(parsed_arguments.input)
  features = {}
  required_features = architecture.auxiliary_features + architecture.feature_predictions
  for feature_prediction in required_features:
    exr_loaded = False

    if feature_prediction.load_data:
      for exr_file in exr_files:
        if feature_prediction.name in exr_file:
          image = OpenEXRDirectory._load_exr(exr_file)
          
          # HACK: Assume just one source input!
          features[Naming.source_feature_name(feature_prediction.name, index=0)] = image
          exr_loaded = True
          
          if height == None:
            height = image.shape[0]
            width = image.shape[1]
          else:
            assert height == image.shape[0]
            assert width == image.shape[1]
          break

    else:
      image = tf.ones([height, width, feature_prediction.number_of_channels])
      if feature_prediction.feature_prediction_type != FeaturePredictionType.COLOR:
        # Direct and indirect need to be 0.5.
        image = tf.scalar_mul(0.5, image)
      features[Naming.source_feature_name(feature_prediction.name, index=0)] = image
      exr_loaded = True
    
    if not exr_loaded:
      # TODO: Improve (DeepBlender)
      raise Exception('Image for \'' + feature_prediction.name + '\' could not be loaded or does not exist.')


  smaller_side_length = min(height, width)
  if smaller_side_length < 16:
    raise Exception('The image needs to have at least a side length of 16 pixels.')
  
  if smaller_side_length < tile_size:
    ratio = tile_overlap_size / tile_size
    tile_size = smaller_side_length
    tile_overlap_size = int(tile_size * ratio)


  # Split the images into tiles.
  iteration_delta = tile_size - (2 * tile_overlap_size)

  width_count = width - (2 * tile_overlap_size) - (2 * iteration_delta)
  width_count = width_count / iteration_delta
  width_count = math.ceil(width_count) + 2

  height_count = height - (2 * tile_overlap_size) - (2 * iteration_delta)
  height_count = height_count / iteration_delta
  height_count = math.ceil(height_count) + 2

  tiled_features_grid = [[None for _ in range(width_count) ] for _ in range(height_count)]

  for height_index in range(height_count):
    if height_index == 0:
      lower_height = 0
      upper_height = tile_size
    elif height_index == height_count - 1:
      upper_height = height
      lower_height = upper_height - tile_size
    else:
      lower_height = height_index * iteration_delta
      upper_height = lower_height + tile_size
    
    for width_index in range(width_count):
      if width_index == 0:
        lower_width = 0
        upper_width = tile_size
      elif width_index == width_count - 1:
        upper_width = width
        lower_width = upper_width - tile_size
      else:
        lower_width = width_index * iteration_delta
        upper_width = lower_width + tile_size

      tiled_features = {}
      for feature_name in features:
        feature = features[feature_name]
        tiled_feature = feature[lower_height:upper_height, lower_width:upper_width]
        tiled_features[feature_name] = tiled_feature

      tiled_features_grid[height_index][width_index] = tiled_features
  
  # We don't need the features anymore.
  features = None

  
  # Directly predicting the results by creating a dataset from the tiled features resulted
  # in a huge computational overhead.
  # Converting the tiled features to tfrecords and predicting with a tfrecords dataset
  # is a questionable approach, but it is significantly faster.
  use_tfrecords = True

  if use_tfrecords:
    temporary_tfrecords_filename = './tmp.tfrecords'
    tfrecords_writer =  tf.python_io.TFRecordWriter(temporary_tfrecords_filename)
    for height_index in range(height_count):
      for width_index in range(width_count):
        tiled_features = tiled_features_grid[height_index][width_index]
        serializable_features = {}

        for tiled_feature_name in tiled_features:
          tiled_feature = tiled_features[tiled_feature_name]
          tiled_feature = tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(tiled_feature.tostring())]))
          serializable_features[tiled_feature_name] = tiled_feature

        example = tf.train.Example(features=tf.train.Features(feature=serializable_features))
        tfrecords_writer.write(example.SerializeToString())
    tfrecords_writer.close()

  if use_CPU_only:
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
  
  if use_tfrecords:
    features_loader = []
    required_features = architecture.auxiliary_features + architecture.feature_predictions
    for feature_prediction in required_features:
      features_loader.append(FeatureLoader(feature_prediction))

    tfrecords_files = [os.path.abspath(temporary_tfrecords_filename)]
    batch_size = 1
    threads = 1
    predictions = estimator.predict(input_fn=lambda: 
        input_fn_tfrecords(
            tfrecords_files, features_loader, architecture.feature_flags,
            tile_size, batch_size, threads))
  else:
    tiled_features_list = []
    for height_index in range(height_count):
      for width_index in range(width_count):
        tiled_features = tiled_features_grid[height_index][width_index]
        tiled_features_list.append(tiled_features)

    predictions = estimator.predict(input_fn=lambda:
        slow_direct_input_fn_predict(tiled_features_list, tile_size, tile_size))

  for height_index in range(height_count):
    for width_index in range(width_count):
      tiled_features_grid[height_index][width_index] = next(predictions)

  predictions = {}
  for feature_prediction_tuple in architecture.feature_prediction_tuples:
    for feature_prediction in feature_prediction_tuple.feature_predictions:
      if feature_prediction.load_data:
        horizontal_feature_stripes = []
        for height_index in range(height_count):
          horizontal_feature_elements = []
          for width_index in range(width_count):
            tiled_predictions = tiled_features_grid[height_index][width_index]
            prediction_name = Naming.feature_prediction_name(feature_prediction.name)
            prediction = tiled_predictions[prediction_name]

            lower_height = 0
            upper_height = tile_size
            lower_width = 0
            upper_width = tile_size

            if width_index != 0 and width_index != width_count - 1:
              lower_width = tile_overlap_size
              upper_width = upper_width - tile_overlap_size
            elif width_index == 0 and width_index == width_count - 1:
              pass
            elif width_index == 0:
              upper_width = upper_width - tile_overlap_size
            else:
              assert width_index == width_count - 1
              existing_width = tile_overlap_size + ((width_count - 1) * (tile_size - (2 * tile_overlap_size)))
              remaining_width = width - existing_width
              lower_width = upper_width - remaining_width
            
            if height_index != 0 and height_index != height_count - 1:
              lower_height = tile_overlap_size
              upper_height = upper_height - tile_overlap_size
            elif height_index == 0 and height_index == height_count - 1:
              pass
            elif height_index == 0:
              upper_height = upper_height - tile_overlap_size
            else:
              assert height_index == height_count - 1
              existing_height = tile_overlap_size + ((height_count - 1) * (tile_size - (2 * tile_overlap_size)))
              remaining_height = height - existing_height
              lower_height = upper_height - remaining_height

            prediction = prediction[lower_height:upper_height, lower_width:upper_width]

            horizontal_feature_elements.append(prediction)
          if len(horizontal_feature_elements) > 1:
            horizontal_feature_stripe = np.concatenate(horizontal_feature_elements, 1)
          else:
            horizontal_feature_stripe = horizontal_feature_elements[0]
          horizontal_feature_stripes.append(horizontal_feature_stripe)
        
        if len(horizontal_feature_stripes) > 1:
          prediction = np.concatenate(horizontal_feature_stripes, 0)
        else:
          prediction = horizontal_feature_stripes[0]
        prediction_name = Naming.feature_prediction_name(feature_prediction.name)
        predictions[prediction_name] = prediction

  diffuse_direct = predictions[Naming.feature_prediction_name(RenderPasses.DIFFUSE_DIRECT)]
  diffuse_indirect = predictions[Naming.feature_prediction_name(RenderPasses.DIFFUSE_INDIRECT)]
  diffuse_color = predictions[Naming.feature_prediction_name(RenderPasses.DIFFUSE_COLOR)]
  
  glossy_direct = predictions[Naming.feature_prediction_name(RenderPasses.GLOSSY_DIRECT)]
  glossy_indirect = predictions[Naming.feature_prediction_name(RenderPasses.GLOSSY_INDIRECT)]
  glossy_color = predictions[Naming.feature_prediction_name(RenderPasses.GLOSSY_COLOR)]
  
  subsurface_direct = predictions[Naming.feature_prediction_name(RenderPasses.SUBSURFACE_DIRECT)]
  subsurface_indirect = predictions[Naming.feature_prediction_name(RenderPasses.SUBSURFACE_INDIRECT)]
  subsurface_color = predictions[Naming.feature_prediction_name(RenderPasses.SUBSURFACE_COLOR)]
  
  transmission_direct = predictions[Naming.feature_prediction_name(RenderPasses.TRANSMISSION_DIRECT)]
  transmission_indirect = predictions[Naming.feature_prediction_name(RenderPasses.TRANSMISSION_INDIRECT)]
  transmission_color = predictions[Naming.feature_prediction_name(RenderPasses.TRANSMISSION_COLOR)]
  
  volume_direct = predictions[Naming.feature_prediction_name(RenderPasses.VOLUME_DIRECT)]
  volume_indirect = predictions[Naming.feature_prediction_name(RenderPasses.VOLUME_INDIRECT)]

  environment = predictions[Naming.feature_prediction_name(RenderPasses.ENVIRONMENT)]
  emission = predictions[Naming.feature_prediction_name(RenderPasses.EMISSION)]

  alpha = predictions[Naming.feature_prediction_name(RenderPasses.ALPHA)]


  # Combined features
  diffuse = np.multiply(diffuse_color, np.add(diffuse_direct, diffuse_indirect))
  glossy = np.multiply(glossy_color, np.add(glossy_direct, glossy_indirect))
  subsurface = np.multiply(subsurface_color, np.add(subsurface_direct, subsurface_indirect))
  transmission = np.multiply(transmission_color, np.add(transmission_direct, transmission_indirect))
  
  # Combined image
  image = np.add(diffuse, glossy)
  image = np.add(image, subsurface)
  image = np.add(image, transmission)
  image = np.add(image, volume_direct)
  image = np.add(image, volume_indirect)
  image = np.add(image, environment)
  image = np.add(image, emission)
  
  
  # TODO: Alpha currently ignored for the combined image. (DeepBlender)

  # Store as npy to open in Blender.
  np.save(parsed_arguments.input + '/' + RenderPasses.COMBINED + '.npy', image)

  np.save(parsed_arguments.input + '/' + RenderPasses.DIFFUSE_DIRECT + '.npy', diffuse_direct)
  np.save(parsed_arguments.input + '/' + RenderPasses.DIFFUSE_INDIRECT + '.npy', diffuse_indirect)
  np.save(parsed_arguments.input + '/' + RenderPasses.DIFFUSE_COLOR + '.npy', diffuse_color)

  np.save(parsed_arguments.input + '/' + RenderPasses.GLOSSY_DIRECT + '.npy', glossy_direct)
  np.save(parsed_arguments.input + '/' + RenderPasses.GLOSSY_INDIRECT + '.npy', glossy_indirect)
  np.save(parsed_arguments.input + '/' + RenderPasses.GLOSSY_COLOR + '.npy', glossy_color)

  np.save(parsed_arguments.input + '/' + RenderPasses.SUBSURFACE_DIRECT + '.npy', subsurface_direct)
  np.save(parsed_arguments.input + '/' + RenderPasses.SUBSURFACE_INDIRECT + '.npy', subsurface_indirect)
  np.save(parsed_arguments.input + '/' + RenderPasses.SUBSURFACE_COLOR + '.npy', subsurface_color)

  np.save(parsed_arguments.input + '/' + RenderPasses.TRANSMISSION_DIRECT + '.npy', transmission_direct)
  np.save(parsed_arguments.input + '/' + RenderPasses.TRANSMISSION_INDIRECT + '.npy', transmission_indirect)
  np.save(parsed_arguments.input + '/' + RenderPasses.TRANSMISSION_COLOR + '.npy', transmission_color)

  np.save(parsed_arguments.input + '/' + RenderPasses.VOLUME_DIRECT + '.npy', volume_direct)
  np.save(parsed_arguments.input + '/' + RenderPasses.VOLUME_INDIRECT + '.npy', volume_indirect)

  np.save(parsed_arguments.input + '/' + RenderPasses.ENVIRONMENT + '.npy', environment)
  np.save(parsed_arguments.input + '/' + RenderPasses.EMISSION + '.npy', emission)
  
  np.save(parsed_arguments.input + '/' + RenderPasses.ALPHA + '.npy', alpha)


  # HACK: Temporary output as png. (DeepBlender)
  # image = 255. * image
  # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # cv2.imwrite(parsed_arguments.input + '/combined.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

  if use_tfrecords:
    os.remove(temporary_tfrecords_filename)

if __name__ == '__main__':
  parsed_arguments, unparsed = parser.parse_known_args()
  main(parsed_arguments)