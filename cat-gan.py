from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import utils
import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from functools import partial
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
""" param """
epoch = 100
batch_size = 100
lr = 2e-4
beta1 = 0.5
z_dim = 128
n_critic = 1 #
n_generator = 1
gan_type="cat-gan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
# keep = range(10)
# keep = [1,3,5]
data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,28,28,1])
num_data = 70000
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

test_data = [[], [], [], [], [], [], [], [], [], []]
colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
plt.ion() # enables interactive mode
for i, j in zip(X, Y):
    if len(test_data[j]) < 100:
        test_data[j].append(i)
test_data_list = test_data[0]
for i in range(1,10):
    test_data_list = np.concatenate((test_data_list, test_data[i]))

""" graphs """
# generator = partial(models.generator_m, heads=1)
generator = models.cat_generator
# discriminator = partial(models.cnn_classifier_2,out_dim=len(keep))
encoder = partial(models.cat_discriminator, out_dim = z_dim)
# classifier = partial(models.cat_discriminator,out_dim=10)
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
random_z = tf.random_normal(shape=(batch_size, z_dim),
                       mean=0, stddev=1, dtype=tf.float32)

#marginal entropy
def mar_entropy(y):
    # y1 = F.sum(y, axis=0) / batchsize
    # y2 = F.sum(-y1 * F.log(y1))
    # return y2
    y1 = tf.reduce_mean(y,axis=0)
    y2=tf.reduce_sum(-y1*tf.log(y1))
    return y2

#conditional entropy
def cond_entropy(y):
    # y1 = -y * F.log(y)
    # y2 = F.sum(y1) / batchsize
    # return y2
    y1=-y*tf.log(y)
    y2 = tf.reduce_sum(y1)/batch_size
    return y2

fake = generator(random_z, reuse=False)
r_mean = encoder(real, reuse=False)
f_mean = encoder(fake)
# with tf.variable_scope('cluster_layer', reuse=False):
#     theta_p = tf.get_variable('theta_p')
#     u_p = tf.get_variable('u_p')
#     lambda_p = tf.get_variable('lambda_p')
# real_flatten = tf.reshape(real, [-1, original_dim
r_p = tf.nn.softmax(r_mean)
f_p = tf.nn.softmax(f_mean)

#discriminator loss
d_loss = -1 * (mar_entropy(r_p) - cond_entropy(r_p) + cond_entropy(f_p))  # Equation (7) upper

#generator loss
g_loss = -mar_entropy(f_p) + cond_entropy(f_p)  # Equation (7) lower


# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('discriminator')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
# d_var_for_save = tf.global_variables('classifier')


# optims
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=beta1).minimize(d_loss, var_list=d_var, global_step=global_step)
g_step = optimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=g_var)

""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
# d_saver = tf.train.Saver(var_list=d_var_for_save)
# summary writer
# Send summary statistics to TensorBoard
tf.summary.scalar('d_loss', d_loss)
tf.summary.scalar('g_loss', g_loss)
images_from_g = generator(random_z, training=False)
tf.summary.image('Generator_image', images_from_g, 12)

#predict
embed_mean = encoder(real)
y = tf.nn.softmax(embed_mean)
predicts = tf.argmax(y,axis=1)


merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)

''' initialization '''
sess.run(tf.global_variables_initializer())
# saver.restore(sess, "results/cat-gan-20180402-140657/checkpoint/model.ckpt")
''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch

def sample_once(it):
    rows = 10
    columns = 10
    feed = {random_z: np.random.normal(size=[rows * columns, z_dim])}
    list_of_generators = [images_from_g]  # used for sampling images
    list_of_names = ['it%d-g.jpg' % it]
    save_dir = dir + "/sample_imgs"
    my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
                             list_of_names=list_of_names, save_dir=save_dir)

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
        real_ipt, y = data_pool.batch(['img', 'label'])
        # z_ipt = np.random.normal(size=[batch_size, z_dim])
        _, _ = sess.run([d_step, g_step], feed_dict={real: real_ipt})
        # _ = sess.run([g_step], feed_dict={random_z: z_ipt})

        if it%10 == 0 :
            # real_ipt = (data_pool.batch('img')+1)/2.
            # real_ipt_ = (data_pool_2.batch('img')+1)/2.
            summary = sess.run(merged, feed_dict={real: real_ipt,
                                                #   real_: real_ipt_,
                                                #   real_2:(data_pool_3.batch('img')+1)/2.,
                                                # real_3:(data_pool_4.batch('img')+1)/2.,
                                                  #random_z: np.random.normal(size=[batch_size, z_dim])
                                                  })
            writer.add_summary(summary, it)
        #
        if it%(batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X[:1000]})
            acc = cluster_acc(predict_y, Y[:1000])
            print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])

            plt.clf()
            sample = sess.run(embed_mean, feed_dict={real: test_data_list})
            # X_embedded = tsne.fit_transform(sample)
            X_embedded = TSNE(n_components=2).fit_transform(sample)
            for i in range(10):
                plt.scatter(X_embedded[i * 100:(i + 1) * 100, 0], X_embedded[i * 100:(i + 1) * 100, 1], color=colors[i],
                            label=str(i), s=2)
                plt.draw()
                # if i == 9:
                #     plt.legend(loc='best')
            plt.show()
            # print(b)

            # real_ipt_ = (data_pool_2.batch('img')+1)/2.
            # y1, y2, y3, = sess.run([predict_y_1, predict_y_2, predict_y_3], feed_dict={real_: real_ipt_,
            #                                       real_2:(data_pool_3.batch('img')+1)/2.,
            #                                     real_3:(data_pool_4.batch('img')+1)/2.,})
            # print(y1)
            # print(y2)
            # print(y3)
            # print('-----')

            # for i,j,k in zip(y1,y2,y3):
            #     counts_1 = counts_2 = counts_3 = [0,0,0]
            #     counts_1[i] += 1
            #     counts_2[i] += 1
            #     counts_3[i] += 1
            #     print(counts_1)


    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        # sample_once(it_offset + max_it)
        # print("Save sample images")
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
    # d_saver.save(sess, dir+"/checkpoint/c-model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
