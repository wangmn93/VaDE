from __future__ import division
# from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
from functools import partial
import data_mnist as data
import math
import my_utils

if 0:

    # conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
    # dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # fc = slim.fully_connected
    relu = tf.nn.relu
    fc_relu = partial(fc, activation_fn=relu)

    real = tf.placeholder(tf.float32, shape=[None, 784])
    with tf.variable_scope("encoder", reuse=False):
        y = fc_relu(real, 500, scope='layer1')

    with tf.variable_scope("encoder", reuse=True):
        weights = tf.get_variable('layer1/weights')
        biases = tf.get_variable('layer1/biases')
    print(weights.shape)
    print(biases.shape)
    a = 0

if 0:
    imgs, _, _ = data.mnist_load('MNIST_data')
    a= 0

if 0:
    batch_size = 100
    z_dim = 10
    n_cen = 5
    z = tf.get_variable('z',shape=(batch_size,z_dim))
    theta_p = tf.get_variable('theta_p', shape=(n_cen))
    u_p = tf.get_variable('u_p', shape=(n_cen, z_dim))
    lambda_p = tf.get_variable('lambda_p', shape=(n_cen, z_dim))

    theta_p_tensor = tf.expand_dims(theta_p,0)
    theta_p_tensor = tf.expand_dims(theta_p_tensor, 2)
    theta_p_tensor = tf.tile(theta_p_tensor, [batch_size, 1, 1])
    u_p_tensor = tf.tile(tf.expand_dims(u_p,0),[batch_size,1,1])
    lambda_p_tensor = tf.tile(tf.expand_dims(lambda_p,0),[batch_size,1,1])
    z_tensor = tf.tile(tf.expand_dims(z,1),[1,n_cen,1])
    print 'theta',theta_p_tensor.shape
    print 'u',u_p_tensor.shape
    print 'lambda',lambda_p_tensor.shape
    print 'z',z_tensor.shape

    z_minus_u_sq = -0.5*(tf.square(z_tensor - u_p_tensor)/lambda_p_tensor)
    log_p_c = tf.log(theta_p_tensor)
    second_term = -0.5*tf.log(2*math.pi*lambda_p_tensor)
    p_c_z = tf.exp(log_p_c+tf.reduce_sum(second_term+z_minus_u_sq, axis=2, keep_dims=True))+1e-10

    print 'p_c_z',p_c_z.shape

    gamma = p_c_z/tf.reduce_sum(p_c_z, axis=1, keep_dims=True)

    print 'gamma',gamma.shape

    #cost

# if 0:
    # def vae_loss(x, x_decoded_mean):
    #     Z = T.transpose(K.repeat(z, n_centroid), [0, 2, 1])
    #     z_mean_t = T.transpose(K.repeat(z_mean, n_centroid), [0, 2, 1])
    #     z_log_var_t = T.transpose(K.repeat(z_log_var, n_centroid), [0, 2, 1])
    #     u_tensor3 = T.repeat(u_p.dimshuffle('x', 0, 1), batch_size, axis=0)
    #     lambda_tensor3 = T.repeat(lambda_p.dimshuffle('x', 0, 1), batch_size, axis=0)
    #     theta_tensor3 = theta_p.dimshuffle('x', 'x', 0) * T.ones((batch_size, latent_dim, n_centroid))
    #
    #     p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
    #                          K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10
    #
    #     gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
    #     gamma_t = K.repeat(gamma, latent_dim)
    #
    #     if datatype == 'sigmoid':
    #         loss = alpha * original_dim * objectives.binary_crossentropy(x, x_decoded_mean) \
    #                + K.sum(0.5 * gamma_t * (
    #         latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(
    #             z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
    #                - 0.5 * K.sum(z_log_var + 1, axis=-1) \
    #                - K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x', 0), batch_size, 0)) * gamma, axis=-1) \
    #                + K.sum(K.log(gamma) * gamma, axis=-1)
    #     else:
    #         loss = alpha * original_dim * objectives.mean_squared_error(x, x_decoded_mean) \
    #                + K.sum(0.5 * gamma_t * (
    #         latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(
    #             z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
    #                - 0.5 * K.sum(z_log_var + 1, axis=-1) \
    #                - K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x', 0), batch_size, 0)) * gamma, axis=-1) \
    #                + K.sum(K.log(gamma) * gamma, axis=-1)
    #
    #     return loss

if 0:
    import matplotlib.pyplot as plt
    import numpy as np
    data_pool = my_utils.getFashion_MNISTDatapool(10)
    a,b = data_pool.batch(['img','label'])
    for i in range(len(a)):
        img_2 = np.reshape(a[i], [28, 28])
        plt.imshow(img_2, cmap='gray')
        plt.show()

if 0:
    # import utils
    # def unpickle(file):
    #     import cPickle
    #     with open(file, 'rb') as fo:
    #         dict = cPickle.load(fo)
    #     return dict
    #
    # def load_full_cifar_10():
    #     img =  None
    #     label = None
    #     for i in range(1,5):
    #         if i ==1:
    #             batch = unpickle('cifar-10-batches-py/data_batch_%d'%i)
    #             img = batch['data']
    #             label = batch['labels']
    #         else:
    #             batch = unpickle('cifar-10-batches-py/data_batch_%d' % i)
    #             img = np.concatenate((img,batch['data']))
    #             label = np.concatenate((label, batch['labels']))
    #     batch = unpickle('cifar-10-batches-py/test_batch')
    #     img = np.concatenate((img, batch['data']))
    #     label = np.concatenate((label, batch['labels']))
    #     return img/255., label
    #
    # def get_FullCifar10Datapool(batch_size):
    #     imgs, labels = load_full_cifar_10()
    #     data_pool = utils.MemoryData({'img': imgs, 'label': labels},
    #                                  batch_size)
    #     return data_pool

    import matplotlib.pyplot as plt
    import numpy as np
    data_pool = my_utils.get_FullCifar10Datapool(5, shift=False) #range -1 ~ 1
    imgs, labels = data_pool.batch()
    for i in range(len(imgs)):
        # img = np.reshape(imgs[i], [3, 32, 32])
        # img = img.transpose([1, 2, 0])
        plt.imshow(imgs[i])
        plt.show()
        print labels[i]
    a=0

if 0:
    import numpy as np
    X, Y = my_utils.load_data('mnist')
    X = np.reshape(X, [70000, 28, 28, 1])
    num_data = 70000
    test_data = [[], [], [], [], [], [], [], [], [], []]
    for i, j in zip(X, Y):
        if len(test_data[j]) < 100:
            test_data[j].append(i)
    import matplotlib.pyplot as plt
    for img in test_data[0]:
        img = np.reshape(img, [28, 28])
        plt.imshow(img, cmap='gray')
        plt.show()


