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
epoch = 200
batch_size = 100
# lr = 1e-3
lr_nn = 0.002
decay_n = 10
decay_factor = 0.9

z_dim = 10
n_centroid = 10
original_dim =784

is_pretrain = True

n_critic = 1 #
n_generator = 1
gan_type="VaDE"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

''' data '''
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
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
    with tf.variable_scope('keras', reuse=False):
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



def gamma_output(z, u_p, theta_p, lambda_p):
    Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])

    u_tensor3 = tf.tile(tf.expand_dims(u_p, [0]), [num_data, 1, 1])
    # u_tensor3 = T.repeat(tf.expand_dims(u_p,[0]), batch_size, axis=0)
    # lambda_tensor3 = T.repeat(tf.expand_dims(lambda_p,[0]), batch_size, axis=0)
    lambda_tensor3 = tf.tile(tf.expand_dims(lambda_p, [0]), [num_data, 1, 1])
    temp_theta_p = tf.expand_dims(theta_p, [0])
    temp_theta_p = tf.expand_dims(temp_theta_p, [0])
    # theta_tensor3 = temp_theta_p * T.ones((batch_size, z_dim, n_centroid))
    theta_tensor3 = tf.tile(temp_theta_p, [num_data, z_dim, 1])

    # @TODO
    # PROBLEM HERE ? add theta z_dim times for each cluster?
    p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                         K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

    gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
    return gamma

def target_distribution(gamma):
    temp = tf.square(gamma)/tf.reduce_sum(gamma,axis=0, keep_dims=True)
    temp = temp/tf.reduce_sum(temp, axis=1, keep_dims=True)
    return temp

def KL(P,Q):
    return tf.reduce_sum(P * tf.log(P/Q), [0,1])

def vae_loss(x, x_decoded_mean, z, z_mean, z_log_var, u_p, theta_p, lambda_p, alpha=1, datatype='sigmoid'):
    Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])
    z_mean_t = tf.transpose(K.repeat(z_mean, n_centroid), [0, 2, 1])
    z_log_var_t = tf.transpose(K.repeat(z_log_var, n_centroid), [0, 2, 1])
    u_tensor3 = tf.tile(tf.expand_dims(u_p, [0]),[batch_size,1,1])
    # u_tensor3 = T.repeat(tf.expand_dims(u_p,[0]), batch_size, axis=0)
    # lambda_tensor3 = T.repeat(tf.expand_dims(lambda_p,[0]), batch_size, axis=0)
    lambda_tensor3 = tf.tile(tf.expand_dims(lambda_p, [0]), [batch_size,1,1])
    temp_theta_p = tf.expand_dims(theta_p, [0])
    temp_theta_p = tf.expand_dims(temp_theta_p, [0])
    # theta_tensor3 = temp_theta_p * T.ones((batch_size, z_dim, n_centroid))
    theta_tensor3 = tf.tile(temp_theta_p, [batch_size, z_dim, 1])

    # @TODO
    # PROBLEM HERE ? add theta z_dim times for each cluster?
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

    # t = target_distribution(gamma)
    return tf.reduce_mean(loss)# + 0.1*KL(t, gamma)

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

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
gmm_var = [var for var in T_vars if var.name.startswith('gmm')]
log_var_var = [var for var in T_vars if var.name.startswith('encoder/log_var_layer')]

#optimizer
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, name='global_step',trainable=False)
vade_step = optimizer(learning_rate=learning_rate, epsilon=1e-4).minimize(loss, var_list=en_var+de_var+gmm_var, global_step=global_step)


""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
# tf.summary.scalar('Total_loss', loss)

tf.summary.image('Real', real, 12)
tf.summary.image('Recon', x_hat, 12)


for i in range(n_centroid):
    # a =u_p[:,i]
    u_p_T = tf.reshape(u_p[:,i],[1, z_dim])
    lambda_p_T = tf.reshape(lambda_p[:,i],[1, z_dim])
    eps = tf.random_normal(shape=(12, z_dim),
                       mean=0, stddev=1, dtype=tf.float32)
    random_z = u_p_T + lambda_p_T * eps
    image = decoder(random_z)
    tf.summary.image('Cluster_%d'%i, image, 12)
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
# load_weight = load_pretrain_weight()
# sess.run(load_weight) #load pretrain weights
ae_saver = tf.train.Saver(var_list=en_var+de_var)
# ae_saver.restore(sess, "results/vae-20180406-172649-current-best/checkpoint/model.ckpt")
# ae_saver.restore(sess, "results/vae-fmnist-20180407-081702-20ep/checkpoint/model.ckpt")
# ae_saver.restore(sess,"results/vae-fmnist-20180409-205638/checkpoint/model.ckpt")
ae_saver.restore(sess, 'results/ae-20180412-153851/checkpoint/model.ckpt') #ep200
def gmm_init():
    # imgs = full_data_pool.batch('img')
    # imgs = (imgs + 1) / 2.

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
        # theta_p.assign(np.ones(n_centroid)/float(n_centroid))
        op_list.append(theta_p.assign(np.ones(n_centroid)/float(n_centroid)))
        op_list.append(u_p.assign(g.means_.T))
        op_list.append(lambda_p.assign(g.covars_.T))
        return tf.group(*(op for op in op_list),name='gmm_init')

#default init
def gmm_init2():
    op_list = []
    with tf.variable_scope('gmm', reuse=True):
        theta_p = tf.get_variable('theta_p')
        u_p = tf.get_variable('u_p')
        lambda_p = tf.get_variable('lambda_p')
        op_list.append(theta_p.assign(np.ones(n_centroid) / float(n_centroid)))
        op_list.append(u_p.assign(np.zeros((z_dim, n_centroid))))
        op_list.append(lambda_p.assign(np.ones((z_dim, n_centroid))))
        return tf.group(*(op for op in op_list), name='gmm_init')


load_gmm = gmm_init()
# load_gmm = gmm_init2()
sess.run(load_gmm) #init gmm params
tf.initialize_variables(log_var_var)

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
        # real_ipt = (real_ipt+1)/2.

        global lr_nn
        #learning rate decay
        if it % (decay_n*batch_epoch) == 0 and it != 0:
            lr_nn = max(lr_nn * decay_factor, 0.0002)
            print('lr: ', lr_nn)

        _ = sess.run([vade_step], feed_dict={real: real_ipt, learning_rate:lr_nn})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        if it % (batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X})
            acc = cluster_acc(predict_y, Y)
            print('full-acc-EPOCH-%d'%(it//(batch_epoch)),acc[0])

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
