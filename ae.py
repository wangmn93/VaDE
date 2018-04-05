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


""" param """
epoch = 100
batch_size = 100
# lr = 1e-3
lr_nn = 0.002
decay_n = 10
decay_factor = 0.9

z_dim = 10
n_centroid = 10
original_dim =784

n_critic = 1 #
n_generator = 1
gan_type="ae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

''' data '''
data_pool = my_utils.getFullMNISTDatapool(batch_size) #range -1 ~ 1
full_data_pool = my_utils.getFullMNISTDatapool(70000)
num_data = 70000
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

""" graphs """
encoder = partial(models.encoder, z_dim = z_dim)
decoder = models.decoder
sampleing = models.sampleing
optimizer = tf.train.AdamOptimizer

#gmm params
with tf.variable_scope('gmm', reuse=False):
    tf.get_variable("theta_p", [n_centroid], dtype=tf.float32)
    tf.get_variable("u_p", [z_dim, n_centroid], dtype=tf.float32)
    tf.get_variable("lambda_p", [z_dim, n_centroid], dtype=tf.float32)

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# encoder
z_mean, z_log_var = encoder(real, reuse=False)

#sampleing
z = sampleing(z_mean, z_log_var)

#decoder
x_hat = decoder(z, reuse=False)



def load_weight_for_one_layer(scope, target_layer, src_model, src_layer_index, op_list):
    src_weights = src_model.layers[src_layer_index].get_weights()
    src_w = src_weights[0]
    src_b = src_weights[1]
    with tf.variable_scope(scope, reuse=True):
        tar_w = tf.get_variable(target_layer+'/'+'weights')
        tar_b = tf.get_variable(target_layer + '/' + 'biases')
        op_list.append(tar_w.assign(src_w))
        op_list.append(tar_b.assign(src_b))

def load_pretrain_weight():
    assign_ops = []
    ae = model_from_json(open('pretrain_weights/ae_mnist.json').read())
    ae.load_weights('pretrain_weights/ae_mnist_weights.h5')

    #load encoder
    load_weight_for_one_layer('encoder', 'layer1', ae, 0, assign_ops)
    load_weight_for_one_layer('encoder', 'layer2', ae, 1, assign_ops)
    load_weight_for_one_layer('encoder', 'layer3', ae, 2, assign_ops)
    load_weight_for_one_layer('encoder', 'mean_layer', ae, 3, assign_ops)
    #load decoder
    load_weight_for_one_layer('decoder', 'output_layer', ae, -1, assign_ops)
    load_weight_for_one_layer('decoder', 'layer3', ae, -2, assign_ops)
    load_weight_for_one_layer('decoder', 'layer2', ae, -3, assign_ops)
    load_weight_for_one_layer('decoder', 'layer1', ae, -4, assign_ops)
    return tf.group(*(op for op in assign_ops),name='load_pretrain_weights')

load_weight = load_pretrain_weight()

def gamma_output(z, u_p, theta_p, lambda_p):
    Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])

    u_tensor3 = tf.tile(tf.expand_dims(u_p, [0]), [batch_size, 1, 1])
    # u_tensor3 = T.repeat(tf.expand_dims(u_p,[0]), batch_size, axis=0)
    # lambda_tensor3 = T.repeat(tf.expand_dims(lambda_p,[0]), batch_size, axis=0)
    lambda_tensor3 = tf.tile(tf.expand_dims(lambda_p, [0]), [batch_size, 1, 1])
    temp_theta_p = tf.expand_dims(theta_p, [0])
    temp_theta_p = tf.expand_dims(temp_theta_p, [0])
    # theta_tensor3 = temp_theta_p * T.ones((batch_size, z_dim, n_centroid))
    theta_tensor3 = tf.tile(temp_theta_p, [batch_size, z_dim, 1])

    # @TODO
    # PROBLEM HERE ? add theta z_dim times for each cluster?
    p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                         K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

    gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
    return gamma


def vae_loss(x, x_decoded_mean, z, z_mean, z_log_var, u_p, theta_p, lambda_p, alpha=1, datatype='sigmoid'):
    Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])
    z_mean_t = tf.transpose(K.repeat(z_mean, n_centroid), [0, 2, 1])
    z_log_var_t = tf.transpose(K.repeat(z_log_var, n_centroid), [0, 2, 1])
    u_tensor3 = tf.tile(tf.expand_dims(u_p, [0]),[batch_size,1,1])
    # u_tensor3 = T.repeat(tf.expand_dims(u_p,[0]), batch_size, axis=0)
    # lambda_tensor3 = T.repeat(tf.expand_dims(lambda_p,[0]), batch_size, axis=0)
    lambda_tensor3 = tf.tile(tf.expand_dims(lambda_p, [0]), [batch_size,1,1])
    temp_theta_p = tf.expand_dims(theta_p,[0])
    temp_theta_p = tf.expand_dims(temp_theta_p, [0])
    # theta_tensor3 = temp_theta_p * T.ones((batch_size, z_dim, n_centroid))
    theta_tensor3 = tf.tile(temp_theta_p,[batch_size, z_dim, 1])

    #@TODO
    #PROBLEM HERE ? add theta z_dim times for each cluster?
    p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                         K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

    gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
    gamma_t = K.repeat(gamma, z_dim)

    if datatype == 'sigmoid':
        loss = alpha * original_dim * objectives.binary_crossentropy(x, x_decoded_mean) \
                   + K.sum(0.5 * gamma_t * (
            z_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(
                    z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                   - 0.5 * K.sum(z_log_var + 1, axis=-1) \
                   - K.sum(K.log(K.repeat_elements(tf.expand_dims(theta_p, [0]), batch_size, 0)) * gamma, axis=-1) \
                   + K.sum(K.log(gamma) * gamma, axis=-1)
    else:
        loss = alpha * original_dim * objectives.mean_squared_error(x, x_decoded_mean) \
               + K.sum(0.5 * gamma_t * (
            z_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(
                z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
               - 0.5 * K.sum(z_log_var + 1, axis=-1) \
               - K.sum(K.log(K.repeat_elements(tf.expand_dims(theta_p, [0]), batch_size, 0)) * gamma, axis=-1) \
               + K.sum(K.log(gamma) * gamma, axis=-1)

    return tf.reduce_mean(loss)

#compute loss
with tf.variable_scope('gmm', reuse=True):
    theta_p = tf.get_variable('theta_p')
    u_p = tf.get_variable('u_p')
    lambda_p = tf.get_variable('lambda_p')
real_flatten = tf.reshape(real, [-1, original_dim])
x_hat_flatten = tf.reshape(x_hat, [-1, original_dim])
loss = vae_loss(real_flatten, x_hat_flatten,z, z_mean, z_log_var, u_p, theta_p, lambda_p)
gammas = gamma_output(z,u_p, theta_p, lambda_p)
predicts = tf.argmax(gammas, axis=1)
# print(loss.shape)
# b = sess.run(tar_w)
# a= 1
# real_flatten = tf.reshape(real, [-1, 784])
# x_hat_flatten = tf.reshape(x_hat, [-1, 784])
#
# epsilon = 1e-10
# recon_loss = -tf.reduce_sum(
#     real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
#             axis=1
#         )
# recon_loss = tf.reduce_mean(recon_loss)
# recon_loss = tf.losses.mean_squared_error(labels=real, predictions=x_hat)
# recon_loss = tf.reduce_mean(recon_loss)






# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]

#optimizer
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, name='global_step',trainable=False)
vade_step = optimizer(learning_rate=learning_rate, epsilon=1e-4).minimize(loss, var_list=en_var+de_var+gmm_var, global_step=global_step)


""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
# tf.summary.scalar('Total_loss', loss)
# tf.summary.scalar('D_loss', d_loss)
# tf.summary.scalar('C_loss', c_loss)
# images_form_de = decoder(random_z)
# images_form_c1 = decoder(z1)
# images_form_c2 = decoder(z2)
# # images_form_c3= decoder(z3)
tf.summary.image('Real', real, 12)
tf.summary.image('Recon', x_hat, 12)
# tf.summary.image('Generator_image_c2', images_form_c2, 12)
# tf.summary.image('Generator_image_c3', images_form_c3, 12)

# tf.summary.histogram('mu_1', mu_1)
# tf.summary.histogram('mu_2', mu_2)
# tf.summary.histogram('mu_3', mu_3)

merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())
sess.run(load_weight) #load pretrain weights

import scipy.io as scio
import sys
import gzip
from six.moves import cPickle
def load_data(dataset):
    path = 'dataset/' + dataset + '/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

    if dataset == 'reuters10k':
        data = scio.loadmat(path + 'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()

    if dataset == 'har':
        data = scio.loadmat(path + 'HAR.mat')
        X = data['X']
        X = X.astype('float32')
        Y = data['Y'] - 1
        X = X[:10200]
        Y = Y[:10200]

    return X, Y
def gmm_init():
    # imgs = full_data_pool.batch('img')
    # imgs = (imgs + 1) / 2.
    X, Y = load_data('mnist')
    X = np.reshape(X, [70000,28, 28,1])
    sample = sess.run(z_mean, feed_dict={real:X})
        # GaussianMixture(n_components=n_classes,
        #                 covariance_type=cov_type
    g = mixture.GMM(n_components=n_centroid, covariance_type='diag')
    g.fit(sample)

    op_list = []
    with tf.variable_scope('gmm', reuse=True):
        theta_p = tf.get_variable('theta_p')
        u_p = tf.get_variable('u_p')
        lambda_p = tf.get_variable('lambda_p')
        theta_p.assign(np.ones(n_centroid)/float(n_centroid))
        op_list.append(theta_p.assign(np.ones(n_centroid)/float(n_centroid)))
        op_list.append(u_p.assign(g.means_.T))
        op_list.append(lambda_p.assign(g.covars_.T))
        return tf.group(*(op for op in op_list),name='gmm_init')

load_gmm = gmm_init()
sess.run(load_gmm) #init gmm params

# a,b,c = sess.run([theta_p, u_p, lambda_p])
# gmm_init()
    # u_p.set_value(floatX(g.means_.T))
    # lambda_p.set_value((floatX(g.covars_.T)))

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


def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt, y = data_pool.batch(['img','label'])
        real_ipt = (real_ipt+1)/2.

        global lr_nn
        #learning rate decay
        if it % (decay_n*batch_epoch) == 0 and it != 0:
            lr_nn = max(lr_nn * decay_factor, 0.0002)

        _ = sess.run([vade_step], feed_dict={real: real_ipt, learning_rate:lr_nn})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            # real_ipt = (data_pool.batch('img')+1)/2.
            # z_ipt =  np.random.normal(size=[batch_size, z_dim])
            summary, predict_y = sess.run([merged,predicts], feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
            # if it % (batch_size*batch_epoch) == 0:
            #     X,Y = full_data_pool.batch(['img','label'])
            #     X = (X + 1) / 2.
            #     predict_y = sess.run(predicts, feed_dict={real: X})
            #     acc = cluster_acc(predict_y, Y)
            #     print('full-acc-it%d'%it,acc[0])
            # else:
            # predict_y = sess.run(predicts, feed_dict={real: real_ipt})
            acc = cluster_acc(predict_y, y)
            print('batch-acc-it%d' % it, acc[0])


            #print acc


    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        # sample_once(it_offset + max_it)
        print("Save sample images")
        training(max_it, it_offset + max_it)



total_it = 0
try:
    training(max_it,0)
    a =0
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
