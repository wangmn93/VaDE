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
lr_nn = 2e-4
z_dim = 100
gamma = 1e-2

n_critic = 1 #
n_generator = 1
gan_type="dcgan-vaegan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

''' data '''
data_pool = my_utils.get_FullCifar10Datapool(batch_size, shift=False) # 0

""" graphs """
encoder = partial(models.encoder, z_dim = z_dim)
decoder = models.decoder2
discriminator = models.discriminator
import models_mnist
sampleing = models_mnist.sampleing
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
random_z = tf.placeholder(tf.float32, shape=[None, z_dim])

# encoder
z_mean, z_log_var = encoder(real, reuse=False)

#sampleing
z = sampleing(z_mean, z_log_var)

#decoder
x_hat = decoder(z, reuse=False)

real_flatten = tf.reshape(real, [-1, 3072])
x_hat_flatten = tf.reshape(x_hat, [-1, 3072])

#===================================
# generate
fake = decoder(random_z)

# dicriminate
r_logit, r_feature = discriminator(real, reuse=False)
de_logit, de_feature = discriminator(x_hat)
f_logit, f_feature = discriminator(fake)

#===================================
#d losses
d_r_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(r_logit), r_logit)
d_de_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(de_logit), de_logit)
d_f_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(f_logit), f_logit)
d_loss = d_r_loss + d_f_loss + d_de_loss

#===================================
#g loss
g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(f_logit), f_logit) +\
         tf.losses.sigmoid_cross_entropy(tf.ones_like(de_logit), de_logit)
#===================================
# epsilon = 1e-10
# recon_loss = -tf.reduce_sum(
#     real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
#             axis=1
#         )
# recon_loss = tf.reduce_mean(recon_loss)
recon_loss = tf.losses.mean_squared_error(de_feature, r_feature)
recon_loss = tf.reduce_mean(recon_loss)

#==========================================
# Latent loss
# Kullback Leibler divergence: measure the difference between two distributions
# Here we measure the divergence between the latent distribution and N(0, 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
latent_loss = tf.reduce_mean(latent_loss)

#==========================================
en_loss = recon_loss + latent_loss

de_loss = gamma*recon_loss + g_loss

dis_loss = d_loss
# vae_loss = recon_loss + latent_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
dis_var = [var for var in T_vars if var.name.startswith('discriminator')]



#optimizer
global_step = tf.Variable(0, name='global_step',trainable=False)
# vae_step = optimizer(learning_rate=lr_nn, beta1=0.5).minimize(vae_loss, var_list=de_var)
en_step = optimizer(learning_rate=lr_nn, beta1=0.5).minimize(en_loss, var_list=en_var, global_step=global_step)
de_step = optimizer(learning_rate=lr_nn, beta1=0.5).minimize(de_loss, var_list=de_var)
dis_step = optimizer(learning_rate=lr_nn, beta1=0.5).minimize(dis_loss, var_list=dis_var)


""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('EN_loss', en_loss)
tf.summary.scalar('DE_loss', de_loss)
tf.summary.scalar('D_loss', dis_loss)
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
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _, _, _ = sess.run([en_step, de_step, dis_step], feed_dict={real: real_ipt, random_z:z_ipt})
        # if it>10000:
        #     _, _ = sess.run([c_step, gmm_step], feed_dict={random_z: z_ipt})
        if it%10 == 0 :
            summary = sess.run(merged, feed_dict={real: real_ipt, random_z:z_ipt})
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
