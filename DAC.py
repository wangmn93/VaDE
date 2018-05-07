from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from keras.models import model_from_json
from functools import partial
import data_mnist as data
from sklearn import mixture
import numpy as np
import  theano.tensor as T
from keras import backend as K
import math
from keras import objectives
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def sigmoid_cross_entropy_without_sum(predict ,label):
    epsilon = 1e-5
    loss = -(label * tf.log(epsilon + predict) + (1 - label) * tf.log(epsilon + 1 - predict))
    return loss

batch_size = 1000
learning_rate = 0.001
epochs = 100
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
real = tf.placeholder(tf.float32, [None, 28, 28, 1])
# keep_prob = tf.placeholder(tf.float32)
optimizer = tf.train.RMSPropOptimizer
#=========================
with tf.variable_scope('threshold', reuse=False):
    alpha = tf.get_variable("alpha", [], dtype=tf.float32, initializer=tf.zeros_initializer)

u_alpha = 0.95-alpha
l_alpha = 0.455 + 0.1*alpha

#===================
feature_labels = models.allconvnet_mnist(real, name='encoder', reuse=False)

f_predict = models.allconvnet_mnist(real, name='encoder',training=False)
predicts = tf.argmax(f_predict, axis=1)

#=======================
# def compute_binary_loss(l, u_thres, l_thres):
# binary_loss = 0
similarity_mat = tf.matmul(feature_labels,feature_labels,transpose_b=True)
#     # print('sim mat ',similarity_mat.shape)
# r_mat_1 = tf.to_int32(similarity_mat>=u_alpha)
# r_mat_1 = tf.to_float(r_mat_1)
# # kkk = r_mat_1
# # r_mat_1 += .5*tf.to_float(similarity_mat<u_alpha)
#
# l_temp_1 = r_mat_1*sigmoid_cross_entropy_without_sum(similarity_mat, r_mat_1)
# binary_loss += tf.reduce_sum(l_temp_1)
#
# r_mat_2 = tf.to_int32(similarity_mat<l_alpha)
# r_mat_2 = tf.to_float(r_mat_2)
# l_temp_2 = r_mat_2 * sigmoid_cross_entropy_without_sum(similarity_mat, 1.-r_mat_2)
# binary_loss += tf.reduce_sum(l_temp_2)

def sigmoid_loss_1(real_flatten, x_hat_flatten):
    epsilon = 1e-10
    recon_loss = -(real_flatten * tf.log(epsilon+x_hat_flatten))
    recon_loss = tf.reduce_sum(recon_loss)
    return recon_loss

def sigmoid_loss_0(real_flatten, x_hat_flatten):
    epsilon = 1e-10
    recon_loss = -((1-real_flatten) * tf.log(epsilon+1-x_hat_flatten))
    recon_loss = tf.reduce_sum(recon_loss)
    return recon_loss

sim_flatten = tf.reshape(similarity_mat,shape=[-1])
mask_1 = sim_flatten >= u_alpha
mask_2 = sim_flatten < l_alpha
after_mask_1 = tf.boolean_mask(sim_flatten, mask_1)
after_mask_2 = tf.boolean_mask(sim_flatten, mask_2)
binary_loss1 = sigmoid_loss_1(tf.ones_like(after_mask_1),after_mask_1)
binary_loss2 = sigmoid_loss_0(tf.zeros_like(after_mask_2),after_mask_2)
binary_loss = binary_loss1 + binary_loss2

select_loss = u_alpha - l_alpha

T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
# de_var = [var for var in T_vars if var.name.startswith('threshold')]
global_step = tf.Variable(0, name='global_step',trainable=False)
cluster_step = optimizer(learning_rate=learning_rate).minimize(binary_loss, var_list=en_var, global_step=global_step)
adaptive_step = optimizer(learning_rate=learning_rate).minimize(select_loss, var_list=[alpha])

#===================
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# ae_saver = tf.train.Saver(var_list=en_var)
    # ae_saver.restore(sess, "results/vae-20180406-172649-current-best/checkpoint/model.ckpt")
    # ae_saver.restore(sess, 'results/ae-20180411-193032/checkpoint/model.ckpt')#ep200 0.96
    # ae_saver.restore(sess, 'results/ae-20180412-134727/checkpoint/model.ckpt')#ep100 0.824
    # ae_saver.restore(sess, 'results/ae-20180412-153851/checkpoint/model.ckpt') #ep200
    # ae_saver.restore(sess, 'results/ae-20180412-160207/checkpoint/model.ckpt') #
    # ae_saver.restore(sess, 'results/ae-20180412-173826/checkpoint/model.ckpt') #ep300 0.81
    # ae_saver.restore(sess, 'results/ae-20180412-175908/checkpoint/model.ckpt') #ep300 SGD Momentum 0.94
    # ae_saver.restore(sess, 'results/ae-20180412-190443/checkpoint/model.ckpt') #ep300 SGD Momentum 0.94
    # ae_saver.restore(sess, 'results/ae-20180413-103410/checkpoint/model.ckpt') #ep100 SGD Momentum 0.94
# ae_saver.restore(sess, 'results/ae-20180415-154410/checkpoint/model.ckpt') #ep150 SGD Momentum 0.953
batch_epoch = len(data_pool) // (batch_size)
it = 0
var_grad = tf.gradients(binary_loss, en_var)[0]
for i in range(epochs):
    for j in range(batch_epoch):
        it = batch_epoch*i+j+1
        real_ipt, y = data_pool.batch(['img', 'label'])
        # c_mat = sess.run(cos_mat, feed_dict={real: real_ipt})
        out = sess.run([var_grad,
                        cluster_step,
                        binary_loss
                        # binary_loss1,
                        # binary_loss2,
                        # binary_loss2,
                        # feature_labels,
                        # similarity_mat,
                        # mask_1,
                        # mask_2,
                        # after_mask_1,
                        # after_mask_2
                        ], feed_dict={real: real_ipt})

        # a = 0
        if it%500==0:
            print('loss',out[2])
        # a=0
    _, u_thres, l_thres = sess.run([adaptive_step, u_alpha, l_alpha], feed_dict={})
    print('u_thres',u_thres,'l_thres', l_thres)
    # from sklearn.cluster import KMeans
    # F = sess.run([feature_labels], feed_dict={real: X[:5000], keep_prob: 1.})
    # predict_y = KMeans(n_clusters=10, n_init=20).fit_predict(F[0])
    # acc = my_utils.cluster_acc(predict_y, Y[:5000])
    # print('kmean-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])

    predict_y = sess.run([predicts], feed_dict={real: X[:2000]})
    acc = my_utils.cluster_acc(predict_y[0], Y[:2000])
    print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
    a=0

    fs, u_h, u_l = sess.run([feature_labels, u_alpha, l_alpha], feed_dict={real: real_ipt})

    sim = []
    dis_sim = []
    for count, elem in enumerate(fs[1:]):
        s = np.dot(fs[0, :], elem)
        # b = 0
        if s > u_h and len(sim)<10:
            img = np.reshape(real_ipt[count], [28, 28])
            sim.append(img)
            print('sim', s)
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(np.reshape(real_ipt[0], [28, 28]), cmap='gray')
            axarr[1].imshow(img, cmap='gray')

            # plt.figure()
            # plt.imshow()
            plt.show()

        if s < u_l  and len(dis_sim)<10:
            img = np.reshape(real_ipt[count], [28, 28])
            dis_sim.append(img)
            print('sim score', s)
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(np.reshape(real_ipt[0], [28, 28]), cmap='gray')
            axarr[1].imshow(img, cmap='gray')

            # plt.figure()
            # plt.imshow()
            plt.show()
        # f, axarr = plt.subplots(3, 3)
        # axarr[0,0].imshow(np.reshape(real_ipt[0], [28, 28]), cmap='gray')
        # b = sim[0]
        # axarr[0,1].imshow(sim[0], cmap='gray')
        # axarr[0, 2].imshow(sim[1], cmap='gray')
        # axarr[1, 0].imshow(sim[2], cmap='gray')
        # axarr[1, 1].imshow(sim[3], cmap='gray')
        # axarr[1,2].imshow(sim[4], cmap='gray')
        # axarr[2, 0].imshow(sim[5], cmap='gray')
        # axarr[2, 1].imshow(sim[6], cmap='gray')
        # axarr[2, 2].imshow(sim[7], cmap='gray')
        # plt.show()

        # f, axarr = plt.subplots(3, 3)
        # axarr[0, 0].imshow(np.reshape(real_ipt[0], [28, 28]), cmap='gray')
        # axarr[0, 1].imshow(dis_sim[0], cmap='gray')
        # axarr[0, 2].imshow(dis_sim[1], cmap='gray')
        # axarr[1, 0].imshow(dis_sim[2], cmap='gray')
        # axarr[1, 1].imshow(dis_sim[3], cmap='gray')
        # axarr[1, 2].imshow(dis_sim[4], cmap='gray')
        # axarr[2, 0].imshow(dis_sim[5], cmap='gray')
        # axarr[2, 1].imshow(dis_sim[6], cmap='gray')
        # axarr[2, 2].imshow(dis_sim[7], cmap='gray')
        # plt.show()

sess.close()