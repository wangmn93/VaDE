from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
from functools import partial

#===============model================
import ops
# import tensorflow.contrib.slim as slim
# from ops import bn
# relu = tf.nn.relu
# fc = partial(ops.flatten_fully_connected, activation_fn=None)

# def call_bn(bn, x, test=False, update_batch_stats=True):
#     if not update_batch_stats:
#         return F.batch_normalization(x, bn.gamma, bn.beta)
#     if test:
#         return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var)
#     else:
#         return bn(x)

# def Encoder(x, reuse=True, training=True, update_batch_stats=True, name="encoder", out_dim=10):
#     bn2 = partial(bn, is_training=training, update_batch_stats=update_batch_stats)
#     # fc_bn_relu = partial(fc, activation_fn=relu, normalizer_fn=bn2)
#     init1 = tf.variance_scaling_initializer(scale=0.02, mode='fan_in', distribution='normal')
#     init2 = tf.variance_scaling_initializer(scale=2e-8, mode='fan_in', distribution='normal')
#     with tf.variable_scope(name, reuse=reuse):
#
#         y = relu(bn2(fc(x, 1200, weights_initializer=init1),dim=1200, name='bn1'))
#         y = relu(bn2(fc(y, 1200, weights_initializer=init1),dim=1200, name='bn2'))
#         logits = fc(y,out_dim, weights_initializer=init2)
#         return logits

#==========load data=============
# import sys
# sys.path.append('mnist')
# from load_mnist import *
#=============load mnist================
# whole = load_mnist_whole(PATH='mnist/', scale=1.0 / 128.0, shift=-1.0)
# n_data = len(whole.data)
# # n_class = np.max(whole.label) + 1
# # print('#class ',n_class)
# dim = whole.data.shape[1]
# nearest_dist = np.loadtxt('mnist/10th_neighbor.txt').astype(np.float32)
# avg_dist = np.average(nearest_dist)
# print('avg 10th dist ',np.average(nearest_dist))
#===============load fmnist============
import my_utils
# data,Y = my_utils.loadFullFashion_MNSIT(shift=True)
data,Y = my_utils.loadFullFashion_MNSIT(shift=False)
data = np.reshape(data, [70000,28,28,1])
# whole = Data(data, Y)
# n_data = len(whole.data)
# dim = whole.data.shape[1]
# # nearest_dist = np.load('fmnist-10th-dist.npy')
# nearest_dist = np.load('fmnist-no-shift-10th-dist.npy')
# avg_dist = np.average(nearest_dist)
# print('avg 10th dist ',np.average(nearest_dist))
#===============load svhn================
# import scipy.io as sio
# from sklearn import preprocessing
# train_data = sio.loadmat('../train_32x32.mat')
# X = np.load('svhn-gist.npy')
# # X = preprocessing.scale(X)
# # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# # X = min_max_scaler.fit_transform(X)
# Y = train_data['y']-1
# whole = Data(X, Y)
# n_data = len(whole.data)
# dim = whole.data.shape[1]
# nearest_dist = np.load('svhn-10th-dist.npy')
# avg_dist = np.average(nearest_dist)
# print('avg 10th dist ',avg_dist)
#=============load cifar 10========================
# import my_utils
# _, Y = my_utils.load_full_cifar_10(shift=False)
# Y = Y[35000:]
# data = np.load('cifar10-imagenet-resnet-2.npy')
# data  = np.reshape(data , [15000, 2048])
# whole = Data(data, Y)
# n_data = len(whole.data)
# dim = whole.data.shape[1]
# nearest_dist = np.load('cifar10-10th-dist.npy')
# avg_dist = np.average(nearest_dist)
# print('avg 10th dist ',np.average(nearest_dist))
#========================================
# batch_size = 250
#============useful func===============

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

# def kl(p, q):
#     return tf.reduce_sum(p * tf.log((p + 1e-8) / (q + 1e-8))) / float(batch_size)
#
# def distance(y0, y1):
#     return kl(tf.nn.softmax(y0), tf.nn.softmax(y1))
#
#
# def entropy(p):
#     if p.data.ndim == 2:
#         return - tf.reduce_sum(p * tf.log(p + 1e-8)) / float(batch_size)
#     elif p.data.ndim == 1:
#         return - tf.reduce_sum(p * tf.log(p + 1e-8))
#     else:
#         raise NotImplementedError
#
# #marginal entropy
# def mar_entropy(y):
#     y1 = tf.reduce_mean(y,axis=0)
#     y2=tf.reduce_sum(-y1*tf.log(y1))
#     return y2
#
# #conditional entropy
# def cond_entropy(y):
#     y1=-y*tf.log(y)
#     y2 = tf.reduce_sum(y1)/batch_size
#     return y2

#===============graph=================
real = tf.placeholder(tf.float32, shape=[None, 28,28,1])
# eps_list = tf.placeholder(tf.float32, shape=[None])

#encoder
# logits = Encoder(real, reuse=False)

# p = tf.nn.softmax(logits)

#predicts
import models_mnist as models
logits_ = models.imsatEncoder(real, reuse=False, training=False, update_batch_stats=False)
p_ = tf.nn.softmax(logits_)
predicts = tf.argmax(p_,axis=1)
T_vars = tf.global_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
# #variables
# T_vars = tf.trainable_variables()
# en_var = [var for var in T_vars if var.name.startswith('encoder')]
#
# #loss
# im_loss = -0.1 * (4*mar_entropy(p) - cond_entropy(p)) #ori
#================vat regularization==============
# logits_wo_p = Encoder(real, update_batch_stats=False)
#
# d = tf.random_normal(shape=(batch_size, dim),
#                        mean=0, stddev=1, dtype=tf.float32)
# d_norm = tf.norm(d, ord='euclidean', axis=1, keep_dims=True)
# d = d/d_norm
#
# print('d ',d.shape)
# init_perturbation = real + 10.*d
# # init_perturbation = real + 1e-6*d
# logits_after_p1 = Encoder(init_perturbation, update_batch_stats=False)
#
# kl_loss = distance(logits_wo_p, logits_after_p1)
#
# d_grad = tf.gradients(kl_loss, [d])[0]
# d_grad = tf.stop_gradient(d_grad)
# d_grad = d_grad/tf.norm(d_grad, ord='euclidean', axis=1, keep_dims=True)
#
# print('d_grad ',d_grad.shape)
# eps = 0.25 * tf.expand_dims(eps_list,1)
# logit = tf.stop_gradient(logits_wo_p)
# final_perturbation = real + eps*d_grad
# # final_perturbation = real + 0.25*d_grad
# logits_after_p2 = Encoder(final_perturbation, update_batch_stats=False)
# vat_reg = distance(logits_wo_p, logits_after_p2)
# loss =im_loss + vat_reg

# optimize
# global_step = tf.Variable(0, name='global_step',trainable=False)
# en_step = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9).minimize(loss, var_list=en_var, global_step=global_step)

""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
# saver
saver = tf.train.Saver(max_to_keep=5, var_list=en_var)
# d_saver = tf.train.Saver(var_list=d_var_for_save)
# summary writer
# Send summary statistics to TensorBoard
# import datetime
# dir="results/imsat-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tf.summary.scalar('loss', loss)
# merged = tf.summary.merge_all()
# logdir = dir+"/tensorboard"
# writer = tf.summary.FileWriter(logdir, sess.graph)
# print('Tensorboard dir: '+logdir)


''' initialization '''
sess.run(tf.global_variables_initializer())

''' train '''
# epoch = 50
# batch_epoch = n_data // (batch_size)
# max_it = epoch * batch_epoch



# def training(max_it, it_offset):
#     print("Max iteration: " + str(max_it))
#
#     for it in range(it_offset, it_offset + max_it):
#         # real_ipt, y = data_pool.batch(['img', 'label'])
#         x_u, _, ind = whole.get_(batch_size, need_index=True)
#         nearest_list = nearest_dist[ind]
#         temp = sess.run([en_step], feed_dict={real: x_u, eps_list:nearest_list})
#
#         if it%10 == 0 :
#             summary = sess.run(merged, feed_dict={real: x_u, eps_list:nearest_list})
#             writer.add_summary(summary, it)
#
#
#         if it%(batch_epoch) == 0:
#             x_, y_ = whole.get_(20000, need_index=False)
#             predict_y = sess.run(predicts, feed_dict={real: x_})
#             acc = cluster_acc(predict_y, y_)
#             print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
#
#
# total_it = 0
try:

    # sess.run(tf.global_variables_initializer())
    # training(max_it,0)
    # total_it = sess.run(global_step)
    # print("Total iterations: "+str(total_it))

    #eva
    saver.restore(sess,'results/imsat-20180603-100343/checkpoint/model.ckpt')
    # x_, y_ = whole.get_(20000, need_index=False)
    predict_y = sess.run(predicts, feed_dict={real: data[:20000]})
    # predict_y_ = sess.run(predicts, feed_dict={real: X[35000:]})
    # all_y = np.concatenate((predict_y, y_))
    acc = my_utils.cluster_acc(predict_y, Y[:20000])
    # np.save('gist-0.364',all_y)
    print('full-acc', acc[0])

except Exception, e:
    traceback.print_exc()
finally:

    # save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    # print("Model saved in path: %s" % save_path)
    # print(" [*] Close main session!")
    sess.close()
