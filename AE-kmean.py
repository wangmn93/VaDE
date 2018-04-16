from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from sklearn.manifold import TSNE
from sklearn import mixture
import matplotlib.pyplot as plt
from functools import partial

""" param """
epoch = 300
batch_size = 256
lr = 1e-3
z_dim = 10
n_critic = 1 #
n_generator = 1

X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
num_data = 70000

#prepare data for plot
test_data = [[], [], [], [], [], [], [], [], [], []]
colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
#             0       1       2       3        4          5        6        7         8       9
plt.ion() # enables interactive mode
for i, j in zip(X, Y):
    if len(test_data[j]) < 100:
        test_data[j].append(i)

test_data_list = test_data[0]
for i in range(1,10):
    test_data_list = np.concatenate((test_data_list, test_data[i]))
gan_type="ae"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False)
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)

""" graphs """
encoder = partial(models.encoder, z_dim=z_dim)
decoder = models.decoder
optimizer = tf.train.MomentumOptimizer

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
ae_step = optimizer(learning_rate=lr, momentum=0.9).minimize(recon_loss, var_list=en_var+de_var, global_step=global_step)


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
        real_ipt = data_pool.batch('img')
        # z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([ae_step], feed_dict={real: real_ipt})



        if it%10 == 0 :


            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        if it%(5*batch_epoch) == 0 and it != 0:
            from sklearn.cluster import KMeans

            # imgs = full_data_pool.batch('img')
            # imgs = (imgs + 1) / 2.

            sample = sess.run(z_mean, feed_dict={real: X})
            predict_y = KMeans(n_clusters=10, n_init=20).fit_predict(sample)
            # predict_y = sess.run(predicts, feed_dict={real: X})
            acc = cluster_acc(predict_y, Y)
            print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
            i = 0
            plt.clf()
            sample = sess.run(z_mean, feed_dict={real: test_data_list})
            X_embedded = TSNE(n_components=2).fit_transform(sample)
            for i in range(10):
                plt.scatter(X_embedded[i*100:(i+1)*100, 0], X_embedded[i*100:(i+1)*100, 1], color=colors[i], label=str(i), s=2)
            # for test_d in test_data:
            #     sample = sess.run(z_mean, feed_dict={real: test_d})
            #     # X_embedded = sample
            #     X_embedded = TSNE(n_components=2).fit_transform(sample)
            #     plt.scatter(X_embedded[:,0],X_embedded[:,1],color=colors[i],label=str(i), s=2)
            #     i += 1
                plt.draw()
            # plt.legend(loc='best')
            plt.show()
                # GaussianMixture(n_components=n_classes,
                #                 covariance_type=cov_type
                # g = mixture.GMM(n_components=10, covariance_type='diag')
                # g.fit(sample)
                # a = 0

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
