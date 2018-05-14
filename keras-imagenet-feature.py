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

""" param """
epoch = 50
batch_size = 100
lr = 2e-4
beta1 = 0.5
gan_type="apx-gist"
dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


''' data '''
import utils
#svhn
# X , _ = my_utils.get_svhn()
import scipy.io as sio
train_data = sio.loadmat('../train_32x32.mat')
X = train_data['X']
X = X.transpose([3, 0, 1, 2])
gist_feature = np.load('svhn-gist.npy')
# Y = train_data['y']
data_pool = utils.MemoryData({'img': X, 'label':gist_feature}, batch_size)

#==============Resize_image=================
real = tf.placeholder(tf.float32, shape=[None, 32,32,3])
resize_op = tf.image.resize_images(real,size=[50,50])
sess = tf.Session()
resized_X = sess.run(resize_op, feed_dict={real: X[:30000]})
sess.close()
#=================KERAS=====================
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.applications.vgg19 import preprocess_input
x = preprocess_input(resized_X)
base_model = VGG19(weights='imagenet',input_tensor = Input(shape=(50, 50, 3)),include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
model.summary()
features = model.predict(x)
np.save('svhn-imagenet', features)
print('finish')
a = 0