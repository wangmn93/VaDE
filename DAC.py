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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool_2x2(x, W):
  return tf.nn.conv2d(x, W,
                        strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 1, 1, 1], padding='VALID')

def all_conv(x_image, keep_prob, name='encoder', reuse=True):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_pool1 = weight_variable([2, 2, 32, 32])
    b_pool1 = bias_variable([32])

    W_pool2 = weight_variable([2, 2, 64, 64])
    b_pool2 = bias_variable([64])
    with tf.variable_scope(name, reuse=reuse):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tf.nn.relu(pool_2x2(h_conv1, W_pool1) + b_pool1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = tf.nn.relu(pool_2x2(h_conv2, W_pool2) + b_pool2)

        W_fc1 = weight_variable([1, 1, 64, 1024])
        b_fc1 = bias_variable([1024])

        #h_conv2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(conv2d(h_pool2, W_fc1) + b_fc1)


        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1, 1, 1024, 10])
        b_fc2 = bias_variable([10])

        h_fc3=tf.nn.relu(conv2d(h_fc1_drop, W_fc2) + b_fc2)

        h_pool3 = avg_pool_7x7(h_fc3)

        y_conv = tf.reshape(h_pool3, [-1, 10])

        #restraint layer
        y_max = tf.reduce_max(y_conv,axis=1, keep_dims=True)
        y_out = tf.exp(y_conv-y_max)
        y_out = y_out/tf.norm(y_out, ord=2, axis=1,keep_dims=True)
        return  y_out

def sigmoid_cross_entropy_without_sum(predict ,label):
    epsilon = 1e-5
    loss = -(label * tf.log(epsilon + predict) + (1 - label) * tf.log(epsilon + 1 - predict))
    return loss

batch_size = 1000
learning_rate = 0.009
epochs = 100
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
real = tf.placeholder(tf.float32, [None, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer
#=========================
with tf.variable_scope('threshold', reuse=False):
    alpha = tf.get_variable("alpha", [], dtype=tf.float32, initializer=tf.zeros_initializer)

u_alpha = 0.95-alpha
l_alpha = 0.455 + 0.1*alpha

#===================
feature_labels,_ = models.dc_encoder(real, reuse=False)

def cos_sim(l):
    sim_mat = tf.matmul(l, l, transpose_b=True)
    print('sim mat', sim_mat)
    norm_1 = tf.norm(l, ord=2, axis=1,keep_dims=True)
    norm_mat = tf.matmul(norm_1, norm_1, transpose_b=True)
    print('norm mat', norm_mat)
    sim_mat = sim_mat/norm_mat
    return sim_mat

cos_mat = cos_sim(tf.exp(feature_labels))

def restrain(l):
    #restrain layer
    f_max = tf.reduce_max(l,axis=1, keep_dims=True)
    f1 = l-f_max
    f2 = tf.exp(f1)
    f3 = f2/tf.norm(f2, ord=2, axis=1,keep_dims=True)
    return f3

feature_labels = restrain(feature_labels)
f_predict,_ = models.dc_encoder(real, training=False)
predicts = tf.argmax(restrain(f_predict), axis=1)

#=======================
# def compute_binary_loss(l, u_thres, l_thres):
binary_loss = 0
similarity_mat = tf.matmul(feature_labels,feature_labels,transpose_b=True)
    # print('sim mat ',similarity_mat.shape)
r_mat_1 = tf.to_int32(similarity_mat>=u_alpha)
r_mat_1 = tf.to_float(r_mat_1)
# kkk = r_mat_1
# r_mat_1 += .5*tf.to_float(similarity_mat<u_alpha)

l_temp_1 = r_mat_1*sigmoid_cross_entropy_without_sum(similarity_mat, r_mat_1)
binary_loss += tf.reduce_sum(l_temp_1)

r_mat_2 = tf.to_int32(similarity_mat<l_alpha)
r_mat_2 = tf.to_float(r_mat_2)
l_temp_2 = r_mat_2 * sigmoid_cross_entropy_without_sum(similarity_mat, 1.-r_mat_2)
binary_loss += tf.reduce_sum(l_temp_2)

select_loss = u_alpha - l_alpha

T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
# de_var = [var for var in T_vars if var.name.startswith('threshold')]
global_step = tf.Variable(0, name='global_step',trainable=False)
cluster_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(binary_loss, var_list=en_var, global_step=global_step)
adaptive_step = optimizer(learning_rate=learning_rate).minimize(select_loss, var_list=[alpha])

#===================
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
        c_mat = sess.run(cos_mat, feed_dict={real: real_ipt})
        grad,_,b_loss,f,sim_mat,l_t_1,l_t_2,r_1,r_2 = sess.run([var_grad,
                                                                         cluster_step,
                                                                 binary_loss,
                                                                 feature_labels,
                                                                 #feature_labels2,
                                                                  #  feature_labels3,
                                                                 similarity_mat,l_temp_1,l_temp_2,r_mat_1,r_mat_2], feed_dict={real: real_ipt, keep_prob: .5})
        if it%10==0:
            print('loss',b_loss)
    _, u_thres, l_thres = sess.run([adaptive_step, u_alpha, l_alpha], feed_dict={})
    print('u_thres',u_thres,'l_thres', l_thres)
    from sklearn.cluster import KMeans
    F = sess.run([feature_labels], feed_dict={real: X[:5000], keep_prob: 1.})
    predict_y = KMeans(n_clusters=10, n_init=20).fit_predict(F[0])
    acc = my_utils.cluster_acc(predict_y, Y[:5000])
    print('kmean-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])

    predict_y = sess.run([predicts], feed_dict={real: X[:5000], keep_prob: 1.})
    acc = my_utils.cluster_acc(predict_y[0], Y[:5000])
    print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
    a=0

sess.close()