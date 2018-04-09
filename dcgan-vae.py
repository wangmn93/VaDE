from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import tensorflow as tf
import models_32x32 as models
import datetime
import my_utils
from functools import partial

import numpy as np



""" param """
epoch = 10
batch_size = 64
# lr = 1e-3
lr_nn = 2e-4
# decay_n = 10
# decay_factor = 0.9

z_dim = 2048
# n_centroid = 10
# original_dim =784

is_pretrain = True

n_critic = 1 #
n_generator = 1
gan_type="dcgan-vae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

''' data '''
data_pool = my_utils.get_FullCifar10Datapool(batch_size, shift=True) # -1 ~ 1
# X,Y = my_utils.load_full_cifar_10(shift=True)
# # X, Y = my_utils.load_data('mnist')
# X = np.reshape(X, [70000,28,28,1])
# num_data = 70000
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

""" graphs """
encoder = partial(models.encoder, z_dim = z_dim)
decoder = models.decoder
import models_mnist
sampleing = models_mnist.sampleing
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

# encoder
z_mean, z_log_var = encoder(real, reuse=False)

#sampleing
z = sampleing(z_mean, z_log_var)

#decoder
x_hat = decoder(z, reuse=False)

real_flatten = tf.reshape(real, [-1, 3072])
x_hat_flatten = tf.reshape(x_hat, [-1, 3072])

pixel_loss =  tf.reduce_sum( tf.square( x_hat_flatten - real_flatten ) , 1 )
pixel_loss = pixel_loss / 3072.0
recon_loss = tf.reduce_mean(pixel_loss)
# recon_loss = tf.losses.mean_squared_error(x_hat_flatten,real_flatten)
# recon_loss = tf.reduce_mean(recon_loss)


# Latent loss
# Kullback Leibler divergence: measure the difference between two distributions
# Here we measure the divergence between the latent distribution and N(0, 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
latent_loss = tf.reduce_mean(latent_loss)
alpha = 1
loss = recon_loss + alpha*latent_loss



# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]


#optimizer
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr_nn, beta1=0.5).minimize(loss, var_list=en_var+de_var, global_step=global_step)


""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', loss)
tf.summary.image('Real', real, 12)
def get_sampler(real):
    # encoder
    z_mean_s, z_log_var_s = encoder(real, training=False)
    # sampleing
    z_s = sampleing(z_mean_s, z_log_var_s)
    # decoder
    x_hat_s = decoder(z_s, training=False)
    return x_hat_s

tf.summary.image('Recon', get_sampler(real), 12)

merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())
''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch



def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt, y = data_pool.batch(['img','label'])

        _ = sess.run([vae_step], feed_dict={real: real_ipt})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        # if it % (10*batch_epoch) == 0:
        #     sample = sess.run(z_mean, feed_dict={real: X})
        #     # GaussianMixture(n_components=n_classes,
        #     #                 covariance_type=cov_type
        #     # g = mixture.GMM(n_components=10, covariance_type='diag')
        #     # g.fit(sample)
        #     print('max: ',np.amax(sample))
        #     print('min: ', np.amin(sample))
        #     a = 0

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
