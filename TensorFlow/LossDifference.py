import tensorflow as tf
from enum import Enum

class LossDifferenceEnum(Enum):
  DIFFERENCE = 1
  ABSOLUTE = 2
  SMOOTH_ABSOLUTE = 3
  SQUARED = 4

class LossDifference:
  
  @staticmethod
  def difference(predicted, target, loss_difference):
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
    result = tf.reduce_sum(result, axis=3)
    return result