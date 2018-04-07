from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils


""" param """
epoch = 70
batch_size = 100
lr = 1e-3
z_dim = 10
n_critic = 1 #
n_generator = 1
gan_type="aae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])

''' data '''
data_pool = my_utils.getMNISTDatapool(batch_size, shift=False) #range 0 ~ 1


""" graphs """
encoder = models.encoder
decoder = models.decoder
optimizer = tf.train.AdamOptimizer
discriminator = models.discriminator_for_latent

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
random_z = tf.placeholder(tf.float32, shape=[None, z_dim])


# encoder
z_mu, z_log_var = encoder(real, reuse=False)

#decoder
x_hat = decoder(z_mu, reuse=False)

real_flatten = tf.reshape(real, [-1, 784])
x_hat_flatten = tf.reshape(x_hat, [-1, 784])

epsilon = 1e-10
recon_loss = -tf.reduce_sum(
    real_flatten * tf.log(epsilon+x_hat_flatten) + (1-real_flatten) * tf.log(epsilon+1-x_hat_flatten),
            axis=1
        )
recon_loss = tf.reduce_mean(recon_loss)

#discriminator
r_logit = discriminator(random_z, reuse=False)
f_logit = discriminator(z_mu)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))
loss = recon_loss + g_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]
dis_var = [var for var in T_vars if var.name.startswith('discriminator')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
vae_step = optimizer(learning_rate=lr).minimize(loss, var_list=en_var+de_var, global_step=global_step)
d_step = optimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=dis_var)


""" train """
''' init '''
# session
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession()

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', loss)
tf.summary.scalar('D_loss', d_loss)

tf.summary.image('Generator_image', real, 12)
tf.summary.image('Generator_image_c1', x_hat, 12)


merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

# def sample_once(it):
#     rows = 10
#     columns = 10
#     feed = {sample_from_gmm: sample_gmm(size=rows*columns),
#             sample_from_c1: sample_from_gaussian(mus[0], cov, 12),
#             sample_from_c2: sample_from_gaussian(mus[1], cov, 12),
#             sample_from_c3: sample_from_gaussian(mus[2], cov, 12)}
#     list_of_generators = [images_form_de, images_form_c1, images_form_c2, images_form_c3]  # used for sampling images
#     list_of_names = ['it%d-de.jpg' % it, 'it%d-c1.jpg' % it, 'it%d-c2.jpg' % it,'it%d-c3.jpg' % it,]
#     save_dir = dir + "/sample_imgs"
#     my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
#                              list_of_names=list_of_names, save_dir=save_dir)

# def plot_latent_space():
#     real_ipt = (data_pool.batch('img') + 1) / 2.
#     real_ipt2 = (data_pool.batch('img') + 1) / 2.
#     real_ipt3 = (data_pool.batch('img') + 1) / 2.
#
#     _ = sess.run([z], feed_dict={real: real_ipt})

def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt = data_pool.batch('img')
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([vae_step], feed_dict={real: real_ipt})

        _ = sess.run([d_step], feed_dict={real: real_ipt, random_z:z_ipt})

        if it%(10*batch_epoch) == 0 :
            samples = sess.run([z_mu], feed_dict={real: X})
            a = 0
        #     real_ipt = (data_pool.batch('img')+1)/2.
        #     summary = sess.run(merged, feed_dict={real: real_ipt,
        #                                           sample_from_gmm:sample_gmm(size=batch_size),
        #                                           sample_from_c1:sample_from_gaussian(mus[0],cov,12),
        #                                           sample_from_c2:sample_from_gaussian(mus[1], cov, 12),
        #                                           sample_from_c3:sample_from_gaussian(mus[2], cov, 12)})
        #     writer.add_summary(summary, it)

    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        # sample_once(it_offset + max_it)
        print("Save sample images")
        training(max_it, it_offset + max_it)



total_it = 0
try:
    training(max_it,0)
    total_it = sess.run(global_step)
    print("Total iterations: "+str(total_it))
except Exception, e:
    traceback.print_exc()
finally:
    var = raw_input("Save sample images?")
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
