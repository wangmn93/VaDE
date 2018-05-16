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
lrelu_2 = partial(ops.leak_relu, leak=0.1)

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

def decoder2(z, reuse=True, name="decoder"):
    fc_relu = partial(fc, activation_fn=relu)
    fc_tahn = partial(fc, activation_fn=tf.nn.tanh)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(z, 2000, scope='layer1')
        y = fc_relu(y, 500, scope='layer2')
        y = fc_relu(y, 500,scope='layer3')
        x = fc_tahn(y,784, scope='output_layer')
        x = tf.reshape(x, [-1, 28, 28, 1])
        return x

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

# selective sampling
def ss_generator(z, reuse=True, name="generator", training=True):
    bn = partial(batch_norm, is_training=training)
    # fc_relu = partial(fc, normalizer_fn=None, activation_fn=relu)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = tf.sigmoid(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
        return y

def ss_discriminator(x, reuse=True, name = "discriminator"):
    fc_lrelu = partial(fc, normalizer_fn=None, activation_fn=lrelu)
    with tf.variable_scope(name, reuse=reuse):
        y =  tf.reshape(x, [-1, 784])
        y = fc_lrelu(y, 1024)
        return fc(y, 1)

def ss_generator_m(z, heads=10,reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 1024)
        img_sets = []
        for _ in range(heads):
            out = tf.sigmoid(fc(y, 784))
            out = tf.reshape(out, [-1, 28, 28, 1])
            img_sets.append(out)
        return img_sets

def generator_m(z, heads=10, dim=64, reuse=True, training=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        img_sets = []
        for _ in range(heads):
            img_sets.append(tf.sigmoid(dconv(y, 1, 5, 2)))

        return img_sets

def generator(z, dim=64, reuse=True, training=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 1024)
        y = fc_bn_relu(y, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = tf.sigmoid(dconv(y, 1, 5, 2))
        return y

def discriminator(img, dim=64, reuse=True, training=True, name= 'discriminator'):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit

from ops2 import conv2d,bn,linear,deconv2d
def discriminator2( x, training=True, reuse=True , name="discriminator"):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    with tf.variable_scope(name, reuse=reuse):
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=training, scope='d_bn2'))
        # net = tf.reshape(net, [batch_size, -1])
        # net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=training, scope='d_bn3'))
        net = fc(net,1024)
        net = lrelu(bn(net, is_training=training, scope='d_bn3'))
        out_logit = linear(net, 1, scope='d_fc4')
        # out = tf.nn.sigmoid(out_logit)

        return out_logit

def generator2(z, training=True, reuse=True, name = "generator"):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope(name, reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=training, scope='g_bn2'))
        net = tf.reshape(net, [64, 7, 7, 128])
        net = tf.nn.relu(
                bn(deconv2d(net, [64, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=training,
                   scope='g_bn3'))

        out = tf.nn.sigmoid(deconv2d(net, [64, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

        return out


def generator2_mimic(z, training=True, reuse=True, name="generator"):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope(name, reuse=reuse):
        net = fc_bn_relu(z, 1024)
        net = fc_bn_relu(net, 128 * 7 * 7)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = dconv_bn_relu(net, 64, 4, 2)
        net = dconv_bn_relu(net, 1, 4, 2)
        out = tf.nn.sigmoid(net)
        # net = tf.nn.relu(
        #         bn(deconv2d(net, [64, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=training,
        #            scope='g_bn3'))
        #
        # out = tf.nn.sigmoid(deconv2d(net, [64, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

        return out

def generator_m2(z,heads=10, training=True, reuse=True, name="generator"):

        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope(name, reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=training, scope='g_bn2'))
        net = tf.reshape(net, [64, 7, 7, 128])
        net = tf.nn.relu(
                bn(deconv2d(net, [64, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=training,
                   scope='g_bn3'))

        img_sets = []
        for i in range(heads):
            out = tf.nn.sigmoid(deconv2d(net, [64, 28, 28, 1], 4, 4, 2, 2, name='g_dc4_h'+str(i)))
            img_sets.append(out)

        return img_sets

def generator_m2_32X32(z, heads=10, training=True, reuse=True, name="generator"):

        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope(name, reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 8 * 8, scope='g_fc2'), is_training=training, scope='g_bn2'))
        net = tf.reshape(net, [64, 8, 8, 128])
        net = tf.nn.relu(
                bn(deconv2d(net, [64, 16, 16, 64], 4, 4, 2, 2, name='g_dc3'), is_training=training,
                   scope='g_bn3'))

        img_sets = []
        for i in range(heads):
            out = tf.nn.sigmoid(deconv2d(net, [64, 32, 32, 3], 4, 4, 2, 2, name='g_dc4_h' + str(i)))
            img_sets.append(out)

        return img_sets

def generator_m2_32X32_dc(z, heads=10, training=True, reuse=True, name="generator"):

        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope(name, reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 2048, scope='g_fc1'), is_training=training, scope='g_bn1'))
        net = tf.reshape(net, [64, 2, 2, 512])
        net = tf.nn.relu(
                bn(deconv2d(net, [64, 4, 4, 4*64], 4, 4, 2, 2, name='g_dc2'), is_training=training,
                   scope='g_bn2'))
        net = tf.nn.relu(
            bn(deconv2d(net, [64, 8, 8, 2 * 64], 4, 4, 2, 2, name='g_dc3'), is_training=training,
               scope='g_bn3'))
        net = tf.nn.relu(
            bn(deconv2d(net, [64, 16, 16, 2 * 64], 4, 4, 2, 2, name='g_dc4'), is_training=training,
               scope='g_bn4'))
        img_sets = []
        for i in range(heads):
            out = tf.nn.tanh(deconv2d(net, [64, 32, 32, 3], 4, 4, 2, 2, name='g_dc5_h' + str(i)))
            img_sets.append(out)

        return img_sets

def discriminator2_32X32_dc(x, training=True, reuse=True, name="discriminator"):

        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    with tf.variable_scope(name, reuse=reuse):
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 64*2, 4, 4, 2, 2, name='d_conv2'), is_training=training, scope='d_bn2'))
        net = lrelu(bn(conv2d(net, 64 * 4, 4, 4, 2, 2, name='d_conv3'), is_training=training, scope='d_bn3'))
        net = lrelu(bn(conv2d(net, 64 * 8, 4, 4, 2, 2, name='d_conv4'), is_training=training, scope='d_bn4'))
            # net = tf.reshape(net, [batch_size, -1])
            # net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=training, scope='d_bn3'))
        # net = fc(net, 1024)
        # net = lrelu(bn(net, is_training=training, scope='d_bn5'))
        # net = fc(net, 1024)
        # net = lrelu(bn(net, is_training=training, scope='d_bn3'))
        # out_logit = linear(net, 1, scope='d_fc4')
        # out_logit = linear(tf.reshape(net, [64, -1]), 1, 'd_h4_lin')
        # out_logit = linear(net, 1, scope='d_fc4')
            # out = tf.nn.sigmoid(out_logit)
        out_logit = fc(net, 1)
        return out_logit

def cnn_classifier(x,keep_prob, out_dim=10,name="classifier", reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=1-keep_prob)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=out_dim)
        # y = tf.nn.softmax(logits)
        return logits




# x_image = tf.reshape(x, [-1, 32, 32, 3])
def cnn_classifier2(x, keep_prob,reuse=True, name='classifier'):
    # def weight_variable(shape):
    #     initial = tf.truncated_normal(shape, stddev=0.1)
    #     return tf.Variable(initial)
    #
    # def bias_variable(shape):
    #     initial = tf.constant(0.1, shape=shape)
    #     return tf.Variable(initial)

    # def conv2d(x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope(name, reuse=reuse):
    # Convolutional layer 1

        h_conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

        h_pool1 = max_pool_2x2(h_conv1)

    # Convolutional layer 2


        h_conv2 = tf.layers.conv2d(
        inputs=h_pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # print(h_pool2.shape)
    # Fully connected layer 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

        # W_fc1 = weight_variable([8 * 8 * 64, 1024])
        # b_fc1 = bias_variable([1024])

        h_fc1 = tf.nn.relu(fc(h_pool2_flat,1024))

    # Dropout
    # keep_prob  = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully connected layer 2 (Output layer)
    #     W_fc2 = weight_variable([1024, 10])
    #     b_fc2 = bias_variable([10])

        logits = fc(h_fc1_drop, 10)
        return logits

# def generator(z, dim=64, reuse=True, training=True, name="generator"):
#     bn = partial(batch_norm, is_training=training)
#     dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
#     fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
#
#     with tf.variable_scope(name, reuse=reuse):
#         y = fc_bn_relu(z, 1024)
#         y = fc_bn_relu(y, 7 * 7 * dim * 2)
#         y = tf.reshape(y, [-1, 7, 7, dim * 2])
#         y = dconv_bn_relu(y, dim * 2, 5, 2)
#         img = tf.sigmoid(dconv(y, 1, 5, 2))
#         return img

def cat_generator(z,reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_lrelu(z, 500)
        y = fc_bn_lrelu(y, 500)
        y = fc_bn_lrelu_2(y, 1000)

        y = tf.sigmoid(fc(y, 784))
        y = tf.reshape(y, [-1, 28, 28, 1])
            # out_put_sets.append(y_1)
        return y

def cat_generator2(z,reuse=True, name = "generator", training = True):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    # fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_lrelu(z, 500)
        y = fc_bn_lrelu(y, 500)
        y = fc_bn_lrelu(y, 1000)

        y = tf.sigmoid(fc(y, 960))
        # y = tf.reshape(y, [-1, 28, 28, 1])
            # out_put_sets.append(y_1)
        return y

def cat_discriminator(z, out_dim=10, reuse=True, name = "discriminator", training = True, stddev=0.3):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    # fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y = z + tf.random_normal(shape=tf.shape(z),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 1000)
        y = y + tf.random_normal(shape=tf.shape(y),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 500)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        logits = fc(y, out_dim)
        # y_out = tf.nn.softmax(logits)
        return logits

def cat_discriminator2(z, out_dim=10, reuse=True, name = "discriminator", training = True, stddev=0.3):
    bn = partial(batch_norm, is_training=training)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu_2, biases_initializer=None)
    # fc_bn_lrelu_2 = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        # y = z + tf.random_normal(shape=tf.shape(z),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(z, 1000)
        # y = y + tf.random_normal(shape=tf.shape(y),mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 500)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        y = fc_bn_lrelu(y, 250)
        # y = y + tf.random_normal(shape=tf.shape(y), mean=0, stddev=stddev, dtype=tf.float32)
        logits = fc(y, out_dim)
        # y_out = tf.nn.softmax(logits)
        return logits

def cluster_layer(z, out_c=10,reuse=True, name = "cluster"):
    with tf.variable_scope(name, reuse=reuse):
        y = fc(z, out_c)
        return tf.nn.softmax(y)

def entropy(q):
    return tf.reduce_sum(-q*tf.log(q), axis=1)

def allconvnet_mnist(x, out_dim=10,name="classifier", training=True, reuse=True):
    # w_init = tf.variance_scaling_initializer(scale=2., mode='fan_in')
    bn = partial(batch_norm, is_training=training)
    # conv_relu = partial(conv, activation_fn=relu)#, weights_initializer=w_init)
    conv_bn_relu = partial(conv, activation_fn=relu, normalizer_fn=bn,biases_initializer=None)#, weights_initializer=w_init)
    with tf.variable_scope(name, reuse=reuse):
        # Convolutional Layer #1
        y = conv_bn_relu(x, 64, 3, 1)
        y = conv_bn_relu(y, 64, 3, 1)
        y = conv_bn_relu(y, 64, 3, 2)
        y = conv_bn_relu(y, 128, 3, 1)
        y = conv_bn_relu(y, 128, 3, 1)
        y = conv_bn_relu(y, 128, 3, 2)
        y = conv_bn_relu(y, out_dim, 1, 1)
        # print(y.shape)

        y = tf.reduce_mean(y, [1,2])
        y = bn(y)
        #restraint layer
        y_max = tf.reduce_max(y, axis=1, keep_dims=True)
        y = tf.exp(y - y_max)
        y = y/tf.norm(y, ord=2, axis=1, keep_dims=True)
        return y

def cnn_discriminator(z, out_dim=10, reuse=True, name = "discriminator" , training=True):
    lrelu_1 = partial(ops.leak_relu, leak=0.1)
    conv_lrelu = partial(conv, activation_fn=lrelu_1)
    fc_lrelu = partial(fc, activation_fn=lrelu_1)
    with tf.variable_scope(name, reuse=reuse):
        y = conv_lrelu(z, 32, 5, 1)
        y = tf.layers.max_pooling2d(inputs=y, pool_size=[3, 3], strides=2)
        y = conv_lrelu(y, 64, 3, 1)
        y = conv_lrelu(y, 64, 3, 1)
        y = tf.layers.max_pooling2d(inputs=y, pool_size=[3, 3], strides=2)
        y = conv_lrelu(y, 128, 3, 1)
        y = conv_lrelu(y, 10, 1, 1)
        y = fc_lrelu(y, 128)
        y = fc(y,out_dim)
        return y

def cnn_discriminator_cifar(z, out_dim=10, reuse=True, name = "discriminator" , training=True):
    lrelu_1 = partial(ops.leak_relu, leak=0.1)
    conv_lrelu = partial(conv, activation_fn=lrelu_1)
    fc_lrelu = partial(fc, activation_fn=lrelu_1)
    with tf.variable_scope(name, reuse=reuse):
        y = conv_lrelu(z, 96, 3, 1)
        y = conv_lrelu(y, 96, 3, 1)
        y = conv_lrelu(y, 96, 3, 1)
        y = tf.layers.max_pooling2d(inputs=y, pool_size=[2, 2], strides=2)
        y = conv_lrelu(y, 192, 3, 1)
        y = conv_lrelu(y, 192, 3, 1)
        y = conv_lrelu(y, 192, 3, 1)
        y = tf.layers.max_pooling2d(inputs=y, pool_size=[3, 3], strides=2)
        y = conv_lrelu(y, 192, 3, 1)
        y = conv_lrelu(y, 192, 1, 1)
        y = conv_lrelu(y, out_dim, 1, 1)
        y = tf.reduce_mean(y, [1, 2])
        # y = fc(y,out_dim)
        return y

def cnn_generator(z, reuse = True, name = "generator", training=True):
    lrelu_1 = partial(ops.leak_relu, leak=0.1)
    dconv_lrelu = partial(dconv,  activation_fn=lrelu_1)
    fc_lrelu = partial(fc, activation_fn=lrelu_1)
    conv_lrelu = partial(conv, activation_fn=lrelu_1)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_lrelu(z, 7*7*96)
        y = tf.reshape(y,shape=[-1,7,7,96])
        y = dconv_lrelu(y, 1, 5, 2)
        y = conv_lrelu(y, 64, 5, 1)
        y = dconv_lrelu(y, 1, 5, 2)
        y = conv_lrelu(y, 64, 5, 1)
        y = conv_lrelu(y, 1, 5, 1)

        # out_put_sets.append(y_1)
        return y

def catdiscriminator(img, dim=64, reuse=True, training=True, name= 'discriminator', out_dim=10):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        logit = fc(y, out_dim)
        return logit