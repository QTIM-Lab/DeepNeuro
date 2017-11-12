import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

class batch_norm(object):

    # Taken from DCGAN-tensorflow on Github. In future, rewrite for multi-backend batchnorm.

    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
        
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)


def leaky_relu(x, leak=0.2, backend='tf', name="lrelu"):
    
    if backend == 'tf':
        return tf.maximum(x, leak*x)


def relu(backend='tf'):

    if backend == 'tf':
        return tf.nn.relu


def tanh(backed='tf'):

    if backend == 'tf':
        return tf.nn.tanh


def sigmoid(backend='tf'):
    
    if backend == 'tf':
        return tf.nn.sigmoid


def dense(tensor, output_size, stddev=0.02, bias_start=0.0, with_w=False):

    if backend == 'tf':

        shape = tensor.get_shape().as_list()

        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(tensor, matrix) + bias, matrix, bias
        else:
            return tf.matmul(tensor, matrix) + bias

def reshape(backend='tf'):

    if backed == 'tf':
        return tf.reshape

    return 


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d", backend='tf'):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
          deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
          deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
          return deconv, w, biases
        else:
          return deconv