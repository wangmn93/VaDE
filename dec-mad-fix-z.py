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

""" param """
epoch = 100
batch_size = 64
# lr = 1e-3
lr_nn = 0.002
# decay_n = 10
# decay_factor = 0.9

z_dim = 10
n_centroid = 10
original_dim =784

is_pretrain = True

n_critic = 1 #
n_generator = 1
gan_type="dec-mad-fix-z"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf.set_random_seed(1234)
''' data '''
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
num_data = 70000
plt.ion() # enables interactive mode
test_data_list, numPerClass = my_utils.getTest_data(numPerClass=100)
colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
#             0       1       2       3        4          5        6        7         8       9
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

""" graphs """
encoder = partial(models.encoder, z_dim = z_dim)
decoder = models.decoder
num_heads = 1
generator = partial(models.generator_m2, heads=num_heads)
discriminator = models.ss_discriminator
sampleing = models.sampleing
optimizer = tf.train.AdamOptimizer

with tf.variable_scope('kmean', reuse=False):
    tf.get_variable("u_p", [n_centroid, z_dim], dtype=tf.float32)

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# encoder
z_mean, _ = encoder(real, reuse=False)

#sampleing
# z = sampleing(z_mean, z_log_var)

#decoder
x_hat = decoder(z_mean, reuse=False)

real_flatten = tf.reshape(real, [-1, 784])
x_hat_flatten = tf.reshape(x_hat, [-1, 784])

epsilon = 1e-10
recon_loss = -tf.reduce_sum(
    real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
            axis=1
        )

# recon_loss = tf.losses.mean_squared_error(x_hat_flatten,real_flatten)
recon_loss = tf.reduce_mean(recon_loss)


#=====================
# z = tf.random_normal(shape=(batch_size, z_dim),
#                        mean=0, stddev=1, dtype=tf.float32)
# z_1 = tf.constant(0.001, shape=[batch_size//2, z_dim], dtype=tf.float32)
# z_2 = tf.constant(-0.001, shape=[batch_size//2, z_dim], dtype=tf.float32)
# z = tf.concat([z_1,z_2],axis=0)

z = tf.constant(0.001, shape=[batch_size, z_dim], dtype=tf.float32)
# z =  tf.placeholder(tf.float32, shape=[None, z_dim])
fake_set = generator(z, reuse=False)
fake = tf.concat(fake_set, 0)
r_logit = discriminator(real,reuse=False)
f_logit = discriminator(fake)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
d_loss = d_loss_real + (1./num_heads)*d_loss_fake
# g_loss = 0
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))


def compute_soft_assign(z):
    with tf.variable_scope('kmean', reuse=True):
        theta_p = tf.get_variable('u_p')
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - theta_p), axis=2) / 1.))
    q **= (1. + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q

def target_distribution2(q):
    weight = q ** 1.5 / tf.reduce_sum(q, axis=0)
    return tf.transpose(tf.transpose(weight)/ tf.reduce_sum(q, axis=1))

# def target_distribution(gamma):
#
#     temp = tf.square(gamma)/tf.reduce_sum(gamma,axis=0, keep_dims=True)
#     temp = temp/tf.reduce_sum(temp, axis=1, keep_dims=True)
#     return temp

def KL(P,Q):
    return tf.reduce_sum(P * tf.log(P/Q), [0,1])

q = compute_soft_assign(z_mean)
predicts = tf.argmax(q, axis=1)
print('soft dist: ',q.shape)
t = target_distribution2(q)
print('target dist: ',t.shape)
KL_loss = KL(t, q)
# beta = 0.01
# KL_recon_loss = beta*KL_loss + recon_loss

f_logit_set = []
init_weight = 0.5
weight = tf.placeholder(tf.float32, shape=[])
g_loss = weight*g_loss #weight down real loss
for i in range(len(fake_set)):
    onehot_labels = tf.one_hot(indices=tf.cast(tf.scalar_mul(i, tf.ones(batch_size)), tf.int32), depth=n_centroid)
    f_m, _ = encoder(fake_set[i])
    f_l = compute_soft_assign(f_m)
    g_loss += tf.reduce_mean(objectives.categorical_crossentropy(onehot_labels, f_l))
    # g_loss += tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f_l, onehot_labels=onehot_labels))

#=======================
#feature extractor
# extracted_feature_real = encoder(real,name='extractor', reuse=False)
# extracted_feature_fake = encoder(fake,name='extractor')
# bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
# from mmd import mix_rbf_mmd2
# kernel_loss = mix_rbf_mmd2(extracted_feature_fake, extracted_feature_real, sigmas=bandwidths)
# kernel_loss = tf.sqrt(kernel_loss)



# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
# ext_var = [var for var in T_vars if var.name.startswith('extractor')]
kmean_var = [var for var in T_vars if var.name.startswith('kmean')]

g_var = [var for var in T_vars if var.name.startswith('generator')]
dis_var = [var for var in T_vars if var.name.startswith('discriminator')]


#optimizer
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, name='global_step',trainable=False)
ae_step = optimizer(learning_rate=learning_rate).minimize(recon_loss, var_list=en_var+de_var, global_step=global_step)
kl_step = tf.train.MomentumOptimizer(learning_rate=0.002, momentum=0.9).minimize(KL_loss, var_list=kmean_var+en_var)

d_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=dis_var)
g_step = optimizer(learning_rate=5*0.0002, beta1=0.5).minimize(g_loss, var_list=g_var)
# recon_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(recon_loss, var_list=de_var+en_var)
# kl_recon_step = tf.train.AdamOptimizer(learning_rate=0.002).minimize(KL_recon_loss, var_list=de_var+en_var)
""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('d_loss', d_loss)
tf.summary.scalar('g_loss', g_loss)
# tf.summary.scalar('mmd', kernel_loss)
tf.summary.image('Real', real, 12)
tf.summary.image('Recon', x_hat, 12)

# image = generator(z, training=False)
# tf.summary.image('G', image, 12)
image_sets = generator(z, training= False)
for img_set in image_sets:
    tf.summary.image('G_images', img_set, 12)
# for i in range(n_centroid):
#     # a =u_p[:,i]
#     u_p_T = tf.reshape(u_p[:,i],[1, z_dim])
#     lambda_p_T = tf.reshape(lambda_p[:,i],[1, z_dim])
#     eps = tf.random_normal(shape=(12, z_dim),
#                        mean=0, stddev=1, dtype=tf.float32)
#     random_z = u_p_T + lambda_p_T * eps
#     image = decoder(random_z)
#     tf.summary.image('Cluster_%d'%i, image, 12)
# tf.summary.image('Generator_image_c3', images_form_c3, 12)

# tf.summary.histogram('mu_1', mu_1)
# tf.summary.histogram('mu_2', mu_2)
# tf.summary.histogram('mu_3', mu_3)

merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

# ''' initialization '''
sess.run(tf.global_variables_initializer())
# load_weight = load_pretrain_weight()
# sess.run(load_weight) #load pretrain weights
# ae_saver = tf.train.Saver(var_list=en_var+de_var)
# ae_saver.restore(sess, "results/vae-20180406-172649-current-best/checkpoint/model.ckpt")
# ae_saver.restore(sess, "results/vae-fmnist-20180407-081702-20ep/checkpoint/model.ckpt")
# ae_saver.restore(sess,"results/vae-fmnist-20180409-205638/checkpoint/model.ckpt")
def kmean_init():
    from sklearn.cluster import KMeans

    # imgs = full_data_pool.batch('img')
    # imgs = (imgs + 1) / 2.

    sample = sess.run(z_mean, feed_dict={real:X})
    #random state 1 ->num 5
    #random state 0 ->num 4,9
    kmeans = KMeans(n_clusters=n_centroid, n_init=20, random_state=0).fit(sample)
        # GaussianMixture(n_components=n_classes,
        #                 covariance_type=cov_type
    # g = mixture.GMM(n_components=n_centroid, covariance_type='diag')
    # g.fit(sample)

    # op_list = []
    with tf.variable_scope('kmean', reuse=True):
        # theta_p = tf.get_variable('theta_p')
        u_p = tf.get_variable('u_p')
        # lambda_p = tf.get_variable('lambda_p')
        # theta_p.assign(np.ones(n_centroid)/float(n_centroid))
        # op_list.append(theta_p.assign(np.ones(n_centroid)/float(n_centroid)))
        # op_list.append(u_p.assign(kmeans.cluster_centers_))
        # op_list.append(lambda_p.assign(g.covars_.T))
        return u_p.assign(kmeans.cluster_centers_)

# #default init
# def gmm_init2():
#     op_list = []
#     with tf.variable_scope('gmm', reuse=True):
#         theta_p = tf.get_variable('theta_p')
#         u_p = tf.get_variable('u_p')
#         lambda_p = tf.get_variable('lambda_p')
#         op_list.append(theta_p.assign(np.ones(n_centroid) / float(n_centroid)))
#         op_list.append(u_p.assign(np.zeros((z_dim, n_centroid))))
#         op_list.append(lambda_p.assign(np.ones((z_dim, n_centroid))))
#         return tf.group(*(op for op in op_list), name='gmm_init')


# load_gmm = gmm_init()
# load_gmm = gmm_init2()
# sess.run(load_gmm) #init gmm params
# tf.initialize_variables(log_var_var)

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

# def sample_once(it):
#     rows = 10
#     columns = 10
#     feed = {random_z: np.random.normal(size=[rows*columns, z_dim])}
#     list_of_generators = [images_form_de, images_form_c1, images_form_c2]  # used for sampling images
#     list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it]
#     save_dir = dir + "/sample_imgs"
#     my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
#                              list_of_names=list_of_names, save_dir=save_dir)
#

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def pretrain(epochs):
    print("Pretrain epoch: " + str(epochs))
    # total_it = it_offset + max_it
    for i in range(epoch):
        for j in range(batch_epoch):
            current_it = i*batch_epoch + j +1
            # for i in range(n_critic):
            real_ipt, y = data_pool.batch(['img', 'label'])
            # real_ipt = (real_ipt+1)/2.



            _ = sess.run([ae_step], feed_dict={real: real_ipt, learning_rate: lr_nn})
        # if it>10000:
            #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
            if current_it % 10 == 0:
                summary = sess.run(merged, feed_dict={real: real_ipt})
                writer.add_summary(summary, current_it)
    #     if it % (batch_epoch) == 0:
    #         predict_y = sess.run(predicts, feed_dict={real: X})
    #         acc = cluster_acc(predict_y, Y)
    #         print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
    #
    # var = raw_input("Continue training for %d iterations?" % max_it)
    # if var.lower() == 'y':
    #     # sample_once(it_offset + max_it)
    #     print("Save sample images")
    #     training(max_it, it_offset + max_it)
tsne = TSNE(n_components=2)
def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt, y = data_pool.batch(['img','label'])
        # real_ipt = (real_ipt+1)/2.

        # global lr_nn
        #learning rate decay
        # if it % (decay_n*batch_epoch) == 0 and it != 0:
        #     lr_nn = max(lr_nn * decay_factor, 0.0002)
        #     print('lr: ', lr_nn)

        _ = sess.run([kl_step], feed_dict={real: real_ipt})
        # if it%10 ==0:
        # _ = sess.run([recon_step], feed_dict={real: real_ipt})
            # for _ in range(5):
            #     _ = sess.run([recon_step], feed_dict={real: real_ipt})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            summary = sess.run(merged, feed_dict={real: real_ipt, weight: init_weight})
            writer.add_summary(summary, it)
        if it % (batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X})
            acc = cluster_acc(predict_y, Y)
            print('full-acc-EPOCH-%d'%(it//(batch_epoch)),acc[0])
            plt.clf()
            sample = sess.run(z_mean, feed_dict={real: test_data_list})
            X_embedded = tsne.fit_transform(sample)
            for i in range(10):
                plt.scatter(X_embedded[i * numPerClass:(i + 1) * numPerClass, 0],
                            X_embedded[i * numPerClass:(i + 1) * numPerClass, 1],
                            color=colors[i],
                            label=str(i), s=2)
                # for test_d in test_data:
                #     sample = sess.run(z_mean, feed_dict={real: test_d})
                #     # X_embedded = sample
                #     X_embedded = TSNE(n_components=2).fit_transform(sample)
                #     plt.scatter(X_embedded[:,0],X_embedded[:,1],color=colors[i],label=str(i), s=2)
                #     i += 1
                plt.draw()
            # plt.legend(loc='best')
            plt.show()
#
#     var = raw_input("Continue training for %d iterations?" % max_it)
#     if var.lower() == 'y':
#         # sample_once(it_offset + max_it)
#         print("Save sample images")
#         training(max_it, it_offset + max_it)
def recon_training(max_it, it_offset):
    print("recon iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt, y = data_pool.batch(['img', 'label'])
        # real_ipt = (real_ipt+1)/2.

        # global lr_nn
        # learning rate decay
        # if it % (decay_n*batch_epoch) == 0 and it != 0:
        #     lr_nn = max(lr_nn * decay_factor, 0.0002)
        #     print('lr: ', lr_nn)
        # _ = sess.run([recon_step], feed_dict={real: real_ipt})
        # if it%10 ==0:
        #     _ = sess.run([recon_step], feed_dict={real: real_ipt})
        # for _ in range(5):
        #     _ = sess.run([recon_step], feed_dict={real: real_ipt})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it % 10 == 0:
            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        if it % (batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X})
            acc = cluster_acc(predict_y, Y)
            print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
            plt.clf()
            sample = sess.run(z_mean, feed_dict={real: test_data_list})
            X_embedded = TSNE(n_components=2).fit_transform(sample)
            for i in range(10):
                plt.scatter(X_embedded[i * numPerClass:(i + 1) * numPerClass, 0],
                            X_embedded[i * numPerClass:(i + 1) * numPerClass, 1],
                            color=colors[i],
                            label=str(i), s=2)
                # for test_d in test_data:
                #     sample = sess.run(z_mean, feed_dict={real: test_d})
                #     # X_embedded = sample
                #     X_embedded = TSNE(n_components=2).fit_transform(sample)
                #     plt.scatter(X_embedded[:,0],X_embedded[:,1],color=colors[i],label=str(i), s=2)
                #     i += 1
                plt.draw()
            # plt.legend(loc='best')
            plt.show()

def gan_train(max_it, it_offset):
    print("gan iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        real_ipt, y = data_pool.batch(['img', 'label'])
        # z_ipt = np.random.normal(size=[batch_size, z_dim])
        # z_ipt = np.random.normal(size=[batch_size, z_dim])
        if it%700 ==0 and it != 0:
            # global init_weight
            # init_weight = max(0.00001, init_weight*0.5)
            print('weight: ',init_weight)
        _, _ = sess.run([d_step,g_step], feed_dict={real: real_ipt, weight:init_weight})
        if it % 10 == 0:
            summary = sess.run(merged, feed_dict={real: real_ipt, weight:init_weight})
            writer.add_summary(summary, it)
total_it = 0
try:
    # training(max_it,0)
    # a =0
    # pretrain(300)
    ae_saver = tf.train.Saver(var_list=en_var+de_var)
    # ext_saver = tf.train.Saver(var_list=ext_var)
    # ae_saver.restore(sess, 'results/ae-20180411-193032/checkpoint/model.ckpt')
    # ae_saver.restore(sess, 'results/ae-20180413-103410/checkpoint/model.ckpt') #ep100 SGD Momentum 0.94
    ae_saver.restore(sess, 'results/ae-20180412-134727/checkpoint/model.ckpt')  # ep100 0.824
    # ext_saver.restore(sess, 'results/ae-20180412-134727/checkpoint/model.ckpt')
    # ae_saver.restore(sess,'results/dec-20180418-110857/checkpoint/model.ckpt') #dec trained 0.828
    # ae_saver.restore(sess,'results/dec-20180418-112752/checkpoint/model.ckpt') #dec trained
    # ae_saver.restore(sess, 'results/dec-mad-20180418-114438-0.001/checkpoint/dec-model.ckpt')
    load_kmean = kmean_init()
    sess.run(load_kmean)
    training(3*batch_epoch,0)
    # print("DEC Model saved in path: %s" %ae_saver.save(sess, dir + "/checkpoint/dec-model.ckpt") )
    gan_train(max_it, 3*batch_epoch)
    # recon_training(max_it,0)
    # total_it = sess.run(global_step)
    # print("Total iterations: "+str(total_it))
    # for i in range(1):
    #     real_ipt = (data_pool.batch('img') + 1) / 2.
    #     summary = sess.run(merged, feed_dict={real: real_ipt})
    #     writer.add_summary(summary, i)
except Exception, e:
    traceback.print_exc()
finally:
    # var = raw_input("Save sample images?")
    # if var.lower() == 'y':
    #     sample_once(total_it)
        # rows = 10
        # columns = 10
        # feed = {z: np.random.normal(size=[rows * columns, z_dim]),
        #         z1:np.random.normal(loc=mus[0], scale=vars[0], size=[rows * columns, z_dim]),
        #         z2: np.random.normal(loc=mus[1], scale=vars[1], size=[rows * columns, z_dim]),
        #         z3: np.random.normal(loc=mus[2], scale=vars[2], size=[rows * columns, z_dim]),
        #         z4: np.random.normal(loc=mus[3], scale=vars[3], size=[rows * columns, z_dim])}
        # list_of_generators = [images_form_g, images_form_c1, images_form_c2, images_form_c3, images_form_c4]  # used for sampling images
        # list_of_names = ['g-it%d.jpg'%total_it, 'c1-it%d.jpg'%total_it, 'c2-it%d.jpg'%total_it, 'c3-it%d.jpg'%total_it, 'c4-it%d.jpg'%total_it]
        # save_dir = dir + "/sample_imgs"
        # my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed, list_of_names=list_of_names, save_dir=save_dir)

    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
