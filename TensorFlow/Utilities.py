import tensorflow as tf

def signed_log1p(inputs):
  return tf.multiply(tf.sign(inputs), tf.log1p(tf.abs(inputs)))

def signed_expm1(inputs):
  return tf.multiply(tf.sign(inputs), tf.expm1(tf.abs(inputs)))

def heaviside(inputs):
  result = tf.sign(inputs)
  result = tf.minimum(result, 0.)
  return result