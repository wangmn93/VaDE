from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# import utils
import traceback
import numpy as np
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from functools import partial
# from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
""" param """
epoch = 40
batch_size = 100
lr = 2e-4
beta1 = 0.5
z_dim = 128
n_critic = 1 #
n_generator = 1
gan_type="cat-gan"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
# import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# image_ind = 10
train_data = sio.loadmat('../train_32x32.mat')

# access to the dict
x_train = train_data['X']
y_train = train_data['y']
x_train = x_train / 255.
x_train = x_train.transpose([3, 0, 1, 2])
# show sample
# print len(y_train)
# for i in range(10):
# 	plt.imshow(x_train[i])
# 	plt.show()
# 	print y_train[i]
# data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
import utils
data_pool = utils.MemoryData({'img': x_train, 'label':y_train}, batch_size)
# data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
# X, Y = my_utils.load_data('mnist')
# X = np.reshape(X, [70000,28,28,1])
# num_data = 70000

# test_data = [[], [], [], [], [], [], [], [], [], []]
# colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
# plt.ion() # enables interactive mode
# for i, j in zip(X, Y):
#     if len(test_data[j]) < 100:
#         test_data[j].append(i)
# test_data_list = test_data[0]
# for i in range(1,10):
#     test_data_list = np.concatenate((test_data_list, test_data[i]))

""" graphs """

generator = models.generator
encoder = partial(models.cnn_discriminator_cifar, name='encoder', out_dim = 10)
optimizer = tf.train.AdamOptimizer

# inputs
real = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
random_z = tf.random_normal(shape=(batch_size, z_dim),
                       mean=0, stddev=1, dtype=tf.float32)

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

fake = generator(random_z, reuse=False)
print('fake shape ',fake.shape)
r_mean = encoder(real, reuse=False)
f_mean = encoder(fake)

r_p = tf.nn.softmax(r_mean)
f_p = tf.nn.softmax(f_mean)


weight = tf.placeholder(tf.float32, shape=[])
init_weight = 1.

#discriminator loss
d_loss = -1 * (mar_entropy(r_p) - cond_entropy(r_p) + weight*cond_entropy(f_p))  # Equation (7) upper

#generator loss
g_loss = -mar_entropy(f_p) + cond_entropy(f_p)  # Equation (7) lower


# trainable variables for each network
T_vars = tf.trainable_variables()
d_var = [var for var in T_vars if var.name.startswith('encoder')]
g_var = [var for var in T_vars if var.name.startswith('generator')]


# optimize
global_step = tf.Variable(0, name='global_step',trainable=False)
d_step = optimizer(learning_rate=lr, beta1=beta1).minimize(d_loss, var_list=d_var, global_step=global_step)
g_step = optimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=g_var)


d_reg_loss = d_loss + 0.001 * tf.add_n([ tf.nn.l2_loss(v) for v in d_var])
d_reg_step = optimizer(learning_rate=lr, beta1=beta1).minimize(d_reg_loss, var_list=d_var)

""" train """
''' init '''
# session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# saver
saver = tf.train.Saver(max_to_keep=5)
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

    for it in range(it_offset, it_offset + max_it):
        real_ipt, y = data_pool.batch(['img', 'label'])

        if it//batch_epoch >25:
            #print('stop G')
            _ = sess.run([d_reg_step], feed_dict={real: real_ipt, weight: init_weight})

        else:
            if it%(10*batch_epoch)==0 and it != 0:
                global init_weight
                init_weight = init_weight*0.9
                print('weight', init_weight)
            _, _ = sess.run([d_step, g_step], feed_dict={real: real_ipt, weight: init_weight})




        if it%10 == 0 :

            summary = sess.run(merged, feed_dict={real: real_ipt,
                                                   weight: init_weight
                                                  })
            writer.add_summary(summary, it)
        #
        if it%(batch_epoch) == 0:
            predict_y = sess.run(predicts, feed_dict={real: X[:2000]})
            # predict_y_2 = sess.run(predicts, feed_dict={real: X[35000:]})
            acc = cluster_acc(predict_y, Y[:2000])
            print('full-acc-EPOCH-%d' % (it // (batch_epoch)), acc[0])
            dist = [0] * 10

            t_dist = [0] * 10
            for py, y_ in zip(predict_y, Y[:2000]):
               dist[py] += 1
               t_dist[y_] += 1
            print('true dist: ', np.array(t_dist) / float(len(predict_y)))
            print('pred dist: ', np.array(dist) / float(len(predict_y)))



    var = raw_input("Continue training for %d iterations?" % max_it)
    if var.lower() == 'y':
        # sample_once(it_offset + max_it)
        # print("Save sample images")
        training(max_it, it_offset + max_it)

total_it = 0
try:
    # en_saver = tf.train.Saver(var_list=d_var)
    #en_saver.restore(sess, 'results/ae-dc-20180426-133140/checkpoint/model.ckpt')  # ep5
    #en_saver.restore(sess, 'results/ae-dc-20180426-134158/checkpoint/model.ckpt') # ep5
    #en_saver.restore(sess, 'results/ae-dc-20180426-143609/checkpoint/model.ckpt') # ep55
    #en_saver.restore(sess, 'results/ae-dc-20180426-145357/checkpoint/model.ckpt') # ep90
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
