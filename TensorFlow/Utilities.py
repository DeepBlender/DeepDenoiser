import tensorflow as tf

def signed_log1p(x):
  return tf.multiply(tf.sign(x), tf.log1p(tf.abs(x)))