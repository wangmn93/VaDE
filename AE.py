from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from sklearn import mixture


""" param """
epoch = 30
batch_size = 100
lr = 1e-3
z_dim = 10
n_critic = 1 #
n_generator = 1

X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
num_data = 70000

gan_type="ae-pretrain-mnist-3ep"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
# data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False)
data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)

""" graphs """
encoder = models.encoder
decoder = models.decoder
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# encoder
z_mean, _ = encoder(real, reuse=False)

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

# trainable variables for each network
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
de_var = [var for var in T_vars if var.name.startswith('decoder')]

# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
ae_step = optimizer(learning_rate=lr).minimize(recon_loss, var_list=en_var+de_var, global_step=global_step)


""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('Total_loss', recon_loss)
tf.summary.image('Real', real, 12)
tf.summary.image('Recon', x_hat, 12)


merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())
# ae_saver = tf.train.Saver(var_list=en_var+de_var)
# ae_saver.restore(sess, "pretrain_weights/ae-pretrain-20180406-090422-mnist/checkpoint/model.ckpt")

''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

# def sample_once(it):
#     rows = 10
#     columns = 10
#     list_of_generators = [x_hat]  # used for sampling images
#     list_of_names = ['it%d-de.jpg' % it]
#     save_dir = dir + "/sample_imgs"
#     my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict={},
#                              list_of_names=list_of_names, save_dir=save_dir)


def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))
    # total_it = it_offset + max_it
    for it in range(it_offset, it_offset + max_it):
        # for i in range(n_critic):
        real_ipt = data_pool.batch('img')
        # z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([ae_step], feed_dict={real: real_ipt})



        if it%10 == 0 :


            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        # if it%batch_epoch == 0 and it != 0:
        #
        #         sample = sess.run(z_mean, feed_dict={real: X})
        #         # GaussianMixture(n_components=n_classes,
        #         #                 covariance_type=cov_type
        #         g = mixture.GMM(n_components=10, covariance_type='diag')
        #         g.fit(sample)
        #         a = 0

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
    # var = raw_input("Save sample images?")
    # if var.lower() == 'y':
    #     sample_once(total_it)
    # save checkpoint
    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
