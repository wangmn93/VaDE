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
# epoch = 100
batch_size = 2000
# lr = 1e-3
# lr_nn = 0.002
# decay_n = 10
# decay_factor = 0.9

z_dim = 10
n_centroid = 10
original_dim =784

is_pretrain = True

n_critic = 1 #
n_generator = 1
gan_type="dec-mad-mmd"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tf.set_random_seed(1234)
''' data '''
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
# X, Y = my_utils.load_data('mnist')
# X = np.reshape(X, [70000,28,28,1])
# num_data = 70000
# plt.ion() # enables interactive mode
# test_data_list, numPerClass = my_utils.getTest_data(numPerClass=100)
# colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
#             0       1       2       3        4          5        6        7         8       9
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

""" graphs """
encoder = partial(models.encoder, z_dim = z_dim)
decoder = models.decoder
num_heads = 1
generator = partial(models.generator_m, heads=num_heads)
discriminator = models.ss_discriminator
#sampleing = models.sampleing
optimizer = tf.train.AdamOptimizer

# with tf.variable_scope('kmean', reuse=False):
#     tf.get_variable("u_p", [n_centroid, z_dim], dtype=tf.float32)

# inputs
real = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])

#==========generator==============
z = tf.random_normal(shape=(batch_size, z_dim),
                       mean=0, stddev=1, dtype=tf.float32)
# z =  tf.placeholder(tf.float32, shape=[None, z_dim])
fake_set = generator(z, reuse=False)
fake = tf.concat(fake_set, 0)

# ========encoder==============
z_mean_real, _ = encoder(real, reuse=False)
z_mean_fake, _ = encoder(fake)

#======MMD loss===============
bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
from mmd import mix_rbf_mmd2
kernel_loss = mix_rbf_mmd2(z_mean_fake, z_mean_real, sigmas=bandwidths)
kernel_loss = tf.sqrt(kernel_loss)


#=======================
# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
g_var = [var for var in T_vars if var.name.startswith('generator')]




""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
# tf.summary.scalar('MMD', kernel_loss)
# tf.summary.image('Real', real, 12)
# tf.summary.image('Generator', fake, 12)

# merged = tf.summary.merge_all()
# logdir = dir+"/tensorboard"
# writer = tf.summary.FileWriter(logdir, sess.graph)
# print('Tensorboard dir: '+logdir)

# ''' initialization '''
sess.run(tf.global_variables_initializer())

def sample_once():
    import utils
    # rows = 10
    # columns = 10
    # feed = {random_z: np.random.normal(size=[rows*columns, z_dim])}
    sample_imgs = sess.run(fake)
    # if normalize:
    #     for i in range(len(sample_imgs)):
    sample_imgs = sample_imgs * 2. - 1.
    save_dir = dir + "/sample_imgs"
    utils.mkdir(save_dir + '/')
    # for imgs, name in zip(sample_imgs, list_of_names):
    my_utils.saveSampleImgs(imgs=sample_imgs, full_path=save_dir + "/" + 'sample.jpg', row=50, column=40)


def measure(max_it, it_offset):
    loss_list = []
    # print("gan iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        real_ipt, y = data_pool.batch(['img', 'label'])
        # if it % 10 == 0:
        loss = sess.run(kernel_loss,feed_dict={real: real_ipt})
        loss_list.append(loss)
        print('loss',loss)
        # summary = sess.run(merged, feed_dict={real: real_ipt})
        # writer.add_summary(summary, it)
    loss_list = np.array(loss_list)
    print('avg loss',np.mean(loss_list))
# total_it = 0
try:
    # training(max_it,0)
    # a =0
    # pretrain(300)
    en_saver = tf.train.Saver(var_list=en_var)
    g_saver = tf.train.Saver(var_list=g_var)


    # g_saver.restore(sess,'results/dec-mad-20180418-114438-0.001/checkpoint/model.ckpt') #0.001
    # g_saver.restore(sess,'results/dec-mad-20180418-160317-1/checkpoint/model.ckpt') #1
    # g_saver.restore(sess, 'results/dec-mad-20180418-140638-0.1/checkpoint/model.ckpt') #0.1
    # g_saver.restore(sess, 'results/dec-mad-20180418-164245-0.5/checkpoint/model.ckpt')#0.5
    # g_saver.restore(sess, 'results/dec-mad-20180418-132505-10/checkpoint/model.ckpt') #10


    # g_saver.restore(sess, 'results/dec-mad-20180419-213301-4-9-0.001/checkpoint/model.ckpt') #0.001
    # g_saver.restore(sess,'results/dec-mad-20180420-142211-4-9-1E-8-DECAY-0.2/checkpoint/model.ckpt') #1e-8 decay 0.2
    g_saver.restore(sess, 'results/dec-mad-20180419-195713-4-9-0.5/checkpoint/model.ckpt') #0.5
    # g_saver.restore(sess, 'results/dec-mad-20180420-073652-4-9-0.0001/checkpoint/model.ckpt') #0.0001
    # g_saver.restore(sess, 'results/dec-mad-20180420-091811-4-9-1e-8/checkpoint/model.ckpt') #1e-8 decay 0.5
    measure(1,0)
    sample_once()
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
    # save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    # print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
