from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
from functools import partial

fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)

def encoder(x, reuse=True, name="encoder", z_dim=10):
    fc_relu = partial(fc, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 500, scope='layer1')
        y = fc_relu(y, 500, scope="layer2")
        y = fc_relu(y, 2000, scope="layer3")
        z_mean = fc(y,z_dim, scope='mean_layer')
        z_log_var = fc(y, z_dim, scope='log_var_layer')
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

def decoder2(z, reuse=True, name="decoder"):
    fc_relu = partial(fc, activation_fn=relu)
    fc_sigmoid = partial(fc, activation_fn=tf.nn.sigmoid)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 2000, scope='layer1')
        y = fc_relu(y, 500, scope='layer2')
        y = fc_relu(y, 500,scope='layer3')
        x = fc(y,784, scope='output_layer')
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x

def discriminator_for_latent(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_lrelu(x, 512)
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

