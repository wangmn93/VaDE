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

import ops
import tensorflow as tf
from functools import partial
import tensorflow.contrib.slim as slim

fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
lrelu_2 = partial(ops.leak_relu, leak=0.1)

def Encoder(x, reuse=True, name="discriminator", z_dim=10):
    fc_relu = partial(fc, activation_fn=relu)
    with tf.variable_scope(name, reuse=reuse):
        y = fc_relu(x, 3000, scope='layer1')
        y = fc_relu(y, 2000, scope="layer2")
        y = fc_relu(y, 1000, scope="layer3")
        y = fc_relu(y, 960, scope="layer4")
        return tf.nn.sigmoid(y)


""" param """
epoch = 20
batch_size = 100
lr = 2e-3
# beta1 = 0.5
gan_type="apx-gist"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
import utils
#svhn
# X , _ = my_utils.get_svhn()
import scipy.io as sio
train_data = sio.loadmat('../train_32x32.mat')
X = train_data['X']/255.
# X = X/255.
X = X.transpose([3, 0, 1, 2])
# X = np.array(X)/float(255.)
gist_feature = np.load('svhn-gist.npy')
# Y = train_data['y']
data_pool = utils.MemoryData({'img': X, 'label':gist_feature}, batch_size)

# inputs
real = tf.placeholder(tf.float32, shape=[None, 32,32,3])
target_gist = tf.placeholder(tf.float32, shape=[None, 960])

""" graphs """
import models_mnist as models
encoder = Encoder
optimizer = tf.train.MomentumOptimizer

#encoder
real_logits = encoder(real, reuse=False)
# gist_fit = tf.nn.sigmoid(real_logits)
gist_fit = tf.nn.relu(real_logits)

#variables
T_vars = tf.trainable_variables()
en_var = [var for var in T_vars if var.name.startswith('discriminator')]

#loss
loss = tf.losses.mean_squared_error(labels=target_gist, predictions=gist_fit) + 0.01*tf.add_n([ tf.nn.l2_loss(v) for v in en_var])

# optimize
global_step = tf.Variable(0, name='global_step',trainable=False)
en_step = optimizer(learning_rate=lr, momentum=.9).minimize(loss, var_list=en_var, global_step=global_step)

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
# tf.summary.scalar('G loss', g_loss)
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
        _, l = sess.run([en_step, loss], feed_dict={real: real_ipt, target_gist:y})

        if it%100 == 0 :
            print('loss ', l)
        #     summary = sess.run(merged, feed_dict={real: real_ipt, target_gist:y})
        #     writer.add_summary(summary, it)



total_it = 0
try:
    # training(max_it,0)
    # total_it = sess.run(global_step)
    # print("Total iterations: "+str(total_it))
    # for i in range(7):
    saver.restore(sess, 'results/apx-gist-20180513-142047/checkpoint/model.ckpt')
    gist_extracted = sess.run(gist_fit,{real:X[:1000]})
    np.save('svhn-gist-apx', gist_extracted)

except Exception, e:
    traceback.print_exc()
finally:

    save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()
