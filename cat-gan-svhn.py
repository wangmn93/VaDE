from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# import utils
import traceback
import numpy as np
import tensorflow as tf
# import models_mnist as models
import datetime
import my_utils
from functools import partial
# from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
# # tsne = TSNE(n_components=2)

import ops
# import tensorflow as tf
# from functools import partial
import tensorflow.contrib.slim as slim
relu = tf.nn.relu
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=2e-5, updates_collections=None)

def Encoder(x, reuse=True, training=True, name="encoder", out_dim=10):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, activation_fn=relu, normalizer_fn=bn)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(x, 1200, scope='layer1')
        y = fc_bn_relu(y, 1200, scope="layer2")
        logits = fc(y,out_dim, scope='layer3')
        return logits

def Generator(z, reuse=False, training=True, name="generator", out_dim=960):
    bn = partial(batch_norm, is_training=training)
    fc_bn_relu = partial(fc, activation_fn=relu, normalizer_fn=bn)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 500, scope='layer1')
        y = fc_bn_relu(y, 1000, scope="layer2")
        y = tf.nn.sigmoid(fc(y, out_dim, scope='layer3'))
        return y

""" param """
epoch = 50
batch_size = 100
lr = 2e-4
beta1 = 0.5
gan_type="catgan-svhn"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
import utils
#svhn
import scipy.io as sio
train_data = sio.loadmat('../train_32x32.mat')
X = np.load('svhn-gist.npy')
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = min_max_scaler.fit_transform(X)
Y = train_data['y']
data_pool = utils.MemoryData({'img': X, 'label':Y}, batch_size)

# inputs
real = tf.placeholder(tf.float32, shape=[None, 960])
random_z = tf.random_normal(shape=(batch_size, 128),
                       mean=0, stddev=1, dtype=tf.float32)
##mnist
# X, Y = my_utils.load_data('mnist')
# data_pool = utils.MemoryData({'img': X, 'label':Y}, batch_size)
## X = np.reshape(X, [70000,28,28,1])
# num_data = 70000
#
# # inputs
# real = tf.placeholder(tf.float32, shape=[None, 784])

""" graphs """
# import models_mnist as models
encoder = Encoder
# encoder = partial(models.cat_discriminator, name='encoder')
generator = Generator
# generator = models.cat_generator2
optimizer = tf.train.AdamOptimizer

#marginal entropy
def mar_entropy(y):
    y1 = tf.reduce_mean(y,axis=0)
    y2=tf.reduce_sum(-y1*tf.log(y1))
    return y2

#conditional entropy
def cond_entropy(y):
    y1=-y*tf.log(y)
    y2 = tf.reduce_sum(y1)/batch_size
    return y2

#encoder
real_logits = encoder(real, reuse=False)
p = tf.nn.softmax(real_logits)

fake = generator(random_z,reuse=False)
fake_logits = encoder(fake)
fake_p = tf.nn.softmax(fake_logits)

#predict
logits_ = encoder(real, training=False)
p_ = tf.nn.softmax(logits_)
predicts = tf.argmax(p_,axis=1)

#variables
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('encoder')]
g_var = [var for var in T_vars if var.name.startswith('generator')]
#loss
# loss = -1 * (mar_entropy(p) - cond_entropy(p)) + 0.1 * tf.add_n([ tf.nn.l2_loss(v) for v in en_var])
loss = -1 * (mar_entropy(p) - cond_entropy(p) + cond_entropy(fake_p))# + 0.01 * tf.add_n([ tf.nn.l2_loss(v) for v in en_var])
loss2 = -1 * (mar_entropy(p) - cond_entropy(p) + cond_entropy(fake_p)) + 0.01 * tf.add_n([ tf.nn.l2_loss(v) for v in en_var])
g_loss = -mar_entropy(fake_p) + cond_entropy(fake_p)

# optimize
global_step = tf.Variable(0, name='global_step',trainable=False)
en_step = optimizer(learning_rate=lr, beta1=beta1).minimize(loss, var_list=en_var, global_step=global_step)
en_step2 = optimizer(learning_rate=lr, beta1=beta1).minimize(loss2, var_list=en_var, global_step=global_step)
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
tf.summary.scalar('loss', loss)
tf.summary.scalar('G loss', g_loss)
merged = tf.summary.merge_all()
logdir = dir+"/tensorboard"
writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard dir: '+logdir)


''' initialization '''
sess.run(tf.global_variables_initializer())

''' train '''
batch_epoch = len(data_pool) // (batch_size)
max_it = epoch * batch_epoch
def training(max_it, it_offset):
    print("Max iteration: " + str(max_it))

    for it in range(it_offset, it_offset + max_it):
        real_ipt, y = data_pool.batch(['img', 'label'])
        if it//(batch_epoch) > 25:
            _ = sess.run(en_step2, feed_dict={real: real_ipt})
        else:
            _, _ = sess.run([en_step, g_step], feed_dict={real: real_ipt})

        if it%10 == 0 :
            summary = sess.run(merged, feed_dict={real: real_ipt})
            writer.add_summary(summary, it)
        #
        if it%(batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X[:5000]})
            acc = my_utils.cluster_acc(predict_y, Y[:5000])
            print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])


total_it = 0
try:
    training(max_it,0)
    total_it = sess.run(global_step)
    print("Total iterations: "+str(total_it))
except Exception, e:
    traceback.print_exc()
finally:

    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
