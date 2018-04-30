from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(ops.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)


def encoder(img, z_dim, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('encoder', reuse=reuse):
        y = conv_bn_lrelu(img, dim, 5, 2)
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        y = conv_bn_lrelu(y, dim * 4, 5, 2)
        y = conv_bn_lrelu(y, dim * 8, 5, 2)
        z_mu = fc(y, z_dim)
        z_log_sigma_sq = fc(y, z_dim)
        return z_mu, z_log_sigma_sq

def discriminator(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        y = conv_bn_lrelu(y, dim * 4, 5, 2)
        y = conv_bn_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit, slim.flatten(y) #return logit and feature layer


# def decoder(z, dim=64, channels=3, reuse=True, training=True):
#     bn = partial(batch_norm, is_training=training)
#     dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
#
#     with tf.variable_scope('decoder', reuse=reuse):
#         y = fc(z, 2 * 2 * dim * 8)
#         y = tf.reshape(y, [-1, 2, 2, dim * 8])
#         y = relu(bn(y))
#         y = dconv_bn_relu(y, dim * 4, 5, 2)
#         y = dconv_bn_relu(y, dim * 2, 5, 2)
#         y = dconv_bn_relu(y, dim * 1, 5, 2)
#         img = tf.tanh(dconv(y, channels, 5, 2))
#         return img

def decoder(z, dim=64, channels=3, reuse=True, training=True, name='decoder'):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc(z, 2 * 2 * dim * 8)
        y = tf.reshape(y, [-1, 2, 2, dim * 8])
        y = relu(bn(y))
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.sigmoid(dconv(y, channels, 5, 2))
        return img

# def cnn_discriminator(z, out_dim=10, reuse=True, name = "discriminator" , training=True):
#     lrelu_1 = partial(ops.leak_relu, leak=0.1)
#     conv_lrelu = partial(conv, activation_fn=lrelu_1)
#     fc_lrelu = partial(fc, activation_fn=lrelu_1)
#     with tf.variable_scope(name, reuse=reuse):
#         y = conv_lrelu(z, 32, 5, 1)
#         y = tf.layers.max_pooling2d(inputs=y, pool_size=[3, 3], strides=2)
#         y = conv_lrelu(y, 64, 3, 1)
#         y = conv_lrelu(y, 64, 3, 1)
#         y = tf.layers.max_pooling2d(inputs=y, pool_size=[3, 3], strides=2)
#         y = conv_lrelu(y, 128, 3, 1)
#         y = conv_lrelu(y, 10, 1, 1)
#         y = fc_lrelu(y, 128)
#         y = fc(y,out_dim)
#         return y
