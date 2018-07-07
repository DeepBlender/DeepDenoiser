import tensorflow as tf
import Utilities
from enum import Enum

class LossDifferenceEnum(Enum):
  DIFFERENCE = 1
  ABSOLUTE = 2
  SMOOTH_ABSOLUTE = 3
  SQUARED = 4,
  SMAPE = 5

class LossDifference:
  
  @staticmethod
  def difference(predicted, target, loss_difference, epsilon=1e-4):
    if loss_difference == LossDifferenceEnum.DIFFERENCE:
      result = tf.subtract(predicted, target)
    elif loss_difference == LossDifferenceEnum.ABSOLUTE:
      difference = tf.subtract(predicted, target)
      result = tf.abs(difference)
    elif loss_difference == LossDifferenceEnum.SMOOTH_ABSOLUTE:
      difference = tf.subtract(predicted, target)
      absolute_difference = tf.abs(difference)
      result = tf.where(
          tf.less(absolute_difference, 1),
          tf.scalar_mul(0.5, tf.square(absolute_difference)),
          tf.subtract(absolute_difference, 0.5))
    elif loss_difference == LossDifferenceEnum.SQUARED:
      result = tf.squared_difference(predicted, target)
    elif loss_difference == LossDifferenceEnum.SMAPE:
      # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
      absolute_difference = tf.abs(tf.subtract(predicted, target))
      denominator = tf.add(tf.add(tf.abs(predicted), tf.abs(target)), epsilon)
      result = tf.divide(absolute_difference, denominator)
    result = tf.reduce_sum(result, axis=3)
    return result