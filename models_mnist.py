from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
from functools import partial
import tensorflow.contrib.slim as slim

fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)

def encoder(x, reuse=True, name="encoder", z_dim=10):
    fc_relu = partial(fc, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 500, scope='layer1')
        y = fc_relu(y, 500, scope="layer2")
        y = fc_relu(y, 2000, scope="layer3")
        z_mean = fc(y,z_dim, scope='mean_layer')
        z_log_var = fc(y, z_dim, scope='log_var_layer')
        return z_mean, z_log_var

def dc_encoder(img, dim=64, reuse=True, training=True, name= 'encoder', z_dim=10):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        z_mean = fc(y, z_dim)
        z_log_var = fc(y, z_dim)
        return z_mean, z_log_var

def sampleing(z_mean, z_log_var):
    eps = tf.random_normal(shape=tf.shape(z_log_var),
                           mean=0, stddev=1, dtype=tf.float32)
    z = z_mean + tf.exp(z_log_var/2) * eps
    return z

def decoder(z, reuse=True, name="decoder"):
    fc_relu = partial(fc, activation_fn=relu)
    fc_sigmoid = partial(fc, activation_fn=tf.nn.sigmoid)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 2000, scope='layer1')
        y = fc_relu(y, 500, scope='layer2')
        y = fc_relu(y, 500,scope='layer3')
        x = fc_sigmoid(y,784, scope='output_layer')
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x

def dc_decoder(z, dim=64, reuse=True, training=True, name="decoder"):
        bn = partial(batch_norm, is_training=training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

        with tf.variable_scope(name, reuse=reuse):
            y = fc_bn_relu(z, 1024)
            y = fc_bn_relu(y, 7 * 7 * dim * 2)
            y = tf.reshape(y, [-1, 7, 7, dim * 2])
            y = dconv_bn_relu(y, dim * 2, 5, 2)
            img = tf.sigmoid(dconv(y, 1, 5, 2))
            return img
# def decoder2(z, reuse=True, name="decoder"):
#     fc_relu = partial(fc, activation_fn=relu)
#     fc_sigmoid = partial(fc, activation_fn=tf.nn.sigmoid)
#     with tf.variable_scope(name, reuse=reuse):
#         y = fc_relu(z, 2000, scope='layer1')
#         y = fc_relu(y, 500, scope='layer2')
#         y = fc_relu(y, 500,scope='layer3')
#         x = fc(y,784, scope='output_layer')
#         x = tf.reshape(x, [-1, 28, 28, 1])
#         return x

def discriminator_for_latent(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_lrelu(x, 512)
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

def multi_c_discriminator(x, out_c=11,reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        y = fc_lrelu(y, 1024)
        return fc(y, out_c)

