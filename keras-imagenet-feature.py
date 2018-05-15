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

# """ param """
# epoch = 50
# batch_size = 100
# lr = 2e-4
# beta1 = 0.5
# gan_type="apx-gist"
# dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
#
# ''' data '''
import utils
#================svhn====================
# X , _ = my_utils.get_svhn()
# import scipy.io as sio
# train_data = sio.loadmat('../train_32x32.mat')
# X = train_data['X']
# X = X.transpose([3, 0, 1, 2])
# gist_feature = np.load('svhn-gist.npy')
# Y = train_data['y']
# data_pool = utils.MemoryData({'img': X, 'label':gist_feature}, batch_size)
#================cifar 10===================
# data_pool = my_utils.get_FullCifar10Datapool(batch_size, shift=False) # -1 ~ 1
X, Y = my_utils.load_full_cifar_10(shift=False)
# X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [len(X), 3, 32, 32])
X = X.transpose([0, 2, 3, 1])
num_data = 70000
#==============Resize_image=================
real = tf.placeholder(tf.float32, shape=[None, 32,32,3])
resize_op = tf.image.resize_images(real,size=[224,224])
sess = tf.Session()
resized_X = None
import math
batch_size = 2000
num_batch = int(math.ceil(num_data/float(batch_size)))
for i in range(num_batch):
    start_idx = i*2000
    end_idx =(i+1)*2000
    if end_idx > num_data:
        end_idx = num_batch
    resized_X_temp = sess.run(resize_op, feed_dict={real: X[start_idx:end_idx]})
    if i == 0:
        resized_X = np.copy(resized_X_temp)
    else:
        resized_X = np.concatenate((resized_X,resized_X_temp))
    print(len(resized_X))
sess.close()
print('Finish resize')
#=================KERAS=====================
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input
x = preprocess_input(resized_X)
# base_model = VGG19(weights='imagenet',input_tensor = Input(shape=(50, 50, 3)),include_top=False)
base_model = ResNet50(weights='imagenet',input_tensor = Input(shape=(224, 224, 3)),include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
base_model.summary()
features = model.predict(x)
np.save('cifar10-imagenet-resnet', features)
print('finish')
a = 0