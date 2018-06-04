from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def flatten_fully_connected(inputs,
                            num_outputs,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):

    if inputs.shape.ndims > 2:
        inputs = slim.flatten(inputs)
    return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)


def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv2d_transpose(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
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


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn"):
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    mean = tf.reduce_mean(x, axis)
    var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
    avg_mean = tf.get_variable(
        name=name + "_mean",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
        trainable=False
    )

    avg_var = tf.get_variable(
        name=name + "_var",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections,
        trainable=False
    )

    gamma = tf.get_variable(
        name=name + "_gamma",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections
    )

    beta = tf.get_variable(
        name=name + "_beta",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
    )

    if is_training:
        avg_mean_assign_op = tf.no_op()
        avg_var_assign_op = tf.no_op()
        if update_batch_stats:
            avg_mean_assign_op = tf.assign(
                avg_mean,
                0.99 * avg_mean + (1 - 0.99) * mean)
            avg_var_assign_op = tf.assign(
                avg_var,
                0.99 * avg_var + (n / (n - 1))
                * (1 - 0.99) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            z = (x - mean) / tf.sqrt(1e-6 + var)
    else:
        z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

    return gamma * z + beta