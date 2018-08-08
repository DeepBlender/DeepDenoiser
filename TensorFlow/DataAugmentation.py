import tensorflow as tf
import math

from Conv2dUtilities import Conv2dUtilities
from RenderPasses import RenderPasses

class DataAugmentation:
  
  @staticmethod
  def flip_left_right(inputs, name, flip, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
    
    # Flip
    inputs = tf.cond(flip > 0, lambda: tf.image.flip_left_right(inputs), lambda: inputs)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = tf.cond(flip > 0, lambda: DataAugmentation._flip_screen_space_normals(inputs), lambda: inputs)
    if name == RenderPasses.NORMAL:
      raise Exception('Flipping for normals is not supported.')
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs
  
  @staticmethod
  def _flip_screen_space_normals(inputs):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(inputs, [1, 1, 1], channel_axis)
    screen_space_normal_x = tf.negative(screen_space_normal_x)
    inputs = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], channel_axis)
    return(inputs)
  
  @staticmethod
  def rotate_90(inputs, k, name, data_format='channels_last'):
    
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
    
    # Rotate
    inputs = tf.image.rot90(inputs, k=k)
    if name == RenderPasses.SCREEN_SPACE_NORMAL:
      inputs = DataAugmentation._rotate_90_screen_space_normals(inputs, k)
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs
    
  @staticmethod
  def _rotate_90_screen_space_normals(inputs, k):
    # 'channels_last' format is assumed.
    data_format = 'channels_last'
    
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.split(inputs, [1, 1, 1], channel_axis)
    
    def _rotate_90(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> -y
      # y -> x
      temporary_screen_space_normal_x = screen_space_normal_x
      screen_space_normal_x = tf.negative(screen_space_normal_y)
      screen_space_normal_y = temporary_screen_space_normal_x
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    def _rotate_180(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> -x
      # y -> -y
      screen_space_normal_x = tf.negative(screen_space_normal_x)
      screen_space_normal_y = tf.negative(screen_space_normal_y)
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    def _rotate_270(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z):
      # x -> y
      # y -> -x
      temporary_screen_space_normal_y = screen_space_normal_y
      screen_space_normal_y = tf.negative(screen_space_normal_x)
      screen_space_normal_x = temporary_screen_space_normal_y
      return screen_space_normal_x, screen_space_normal_y, screen_space_normal_z
    
    cases =[
        (tf.equal(k, 1), lambda: _rotate_90(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z)),
        (tf.equal(k, 2), lambda: _rotate_180(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z)),
        (tf.equal(k, 3), lambda: _rotate_270(screen_space_normal_x, screen_space_normal_y, screen_space_normal_z))]
    screen_space_normal_x, screen_space_normal_y, screen_space_normal_z = tf.case(
        cases, default=lambda: (screen_space_normal_x, screen_space_normal_y, screen_space_normal_z), exclusive=True)
    
    inputs = tf.concat([screen_space_normal_x, screen_space_normal_y, screen_space_normal_z], channel_axis)
    return(inputs)
  
  @staticmethod
  def permute_rgb(inputs, permute, data_format='channels_last'):
    assert Conv2dUtilities.has_valid_shape(inputs)
    assert Conv2dUtilities.number_of_channels(inputs, data_format) == 3
    
    def _permute_rgb(inputs, permutation):
      channel_axis = Conv2dUtilities.channel_axis(inputs, data_format)
      result = tf.split(inputs, [1, 1, 1], channel_axis)
      result = tf.concat([result[permutation[0]], result[permutation[1]], result[permutation[2]]], channel_axis)
      return result
    
    cases =[
        (tf.equal(permute, 1), lambda: _permute_rgb(inputs, [0, 2, 1])),
        (tf.equal(permute, 2), lambda: _permute_rgb(inputs, [1, 0, 2])),
        (tf.equal(permute, 3), lambda: _permute_rgb(inputs, [1, 2, 0])),
        (tf.equal(permute, 4), lambda: _permute_rgb(inputs, [2, 0, 1])),
        (tf.equal(permute, 5), lambda: _permute_rgb(inputs, [2, 1, 0]))]
    inputs = tf.case(cases, default=lambda: inputs, exclusive=True)
    
    return inputs
  
  def random_rotation_matrix(random_vector):
    #assert len(random_vector) == 3
    #assert 0. <= random_vector[0] and random_vector[0] <= 1.
    #assert 0. <= random_vector[1] and random_vector[1] <= 1.
    #assert 0. <= random_vector[2] and random_vector[2] <= 1.
    
    # Source: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    two_pi = tf.multiply(2., math.pi)
    
    # Rotation about the pole (Z). 
    theta = tf.multiply(random_vector[0], two_pi)
    
    # For direction of pole deflection.
    phi = tf.multiply(random_vector[1], two_pi)
    
    # For magnitude of pole deflection.
    z = tf.multiply(random_vector[2], 2.)
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if random_vector[1] and random_vector[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    r = tf.sqrt(z)
    Vx = tf.multiply(tf.sin(phi), r)
    Vy = tf.multiply(tf.cos(phi), r)
    Vz = tf.sqrt(tf.subtract(2., z))
    
    # Compute the row vector S = Transpose(V) * R, where R is a simple
    # rotation by theta about the z-axis.  No need to compute Sz since
    # it's just Vz. 
    st = tf.sin(theta)
    ct = tf.cos(theta)
    Sx = tf.subtract(tf.multiply(Vx, ct), tf.multiply(Vy, st))
    Sy = tf.add(tf.multiply(Vx, st), tf.multiply(Vy, ct))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which
    # is equivalent to V S - R.
    result = []
    result.append(tf.subtract(tf.multiply(Vx, Sx), ct))
    result.append(tf.subtract(tf.multiply(Vx, Sy), st))
    result.append(tf.multiply(Vx, Vz))
    
    result.append(tf.add(tf.multiply(Vy, Sx), st))
    result.append(tf.subtract(tf.multiply(Vy, Sy), ct))
    result.append(tf.multiply(Vy, Vz))
    
    result.append(tf.multiply(Vz, Sx))
    result.append(tf.multiply(Vz, Sy))
    
    # This equals Vz * Vz - 1.0
    result.append(tf.subtract(1., z))
    
    result = tf.reshape(result, [3, 3])
    return result
  
  @staticmethod
  def rotate_normal(inputs, rotation_matrix, data_format='channels_last'):
  
    # Convert to 'channels_last' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_last')
  
    height, width = Conv2dUtilities.height_width(inputs, data_format)
    inputs = tf.reshape(inputs, [height * width, 3])
    inputs = tf.matmul(inputs, rotation_matrix)
    inputs = tf.reshape(inputs, [height, width, 3])
    
    # Convert back to 'channels_first' if needed.
    if data_format == 'channels_first':
      inputs = Conv2dUtilities.convert_to_data_format(inputs, 'channels_first')
    
    return inputs


class DataAugmentationUsage:

  def __init__(self, use_rotate_90, use_flip_left_right, use_rgb_permutation, use_normal_rotation):
    self.use_rotate_90 = use_rotate_90
    self.use_flip_left_right = use_flip_left_right
    self.use_rgb_permutation = use_rgb_permutation
    self.use_normal_rotation = use_normal_rotation
