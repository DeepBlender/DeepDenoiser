import tensorflow as tf

def signed_log1p(x):
  return tf.multiply(tf.sign(x), tf.log1p(tf.abs(x)))

def signed_expm1(x):
  return tf.multiply(tf.sign(x), tf.expm1(tf.abs(x)))