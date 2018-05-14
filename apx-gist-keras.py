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
# epoch = 50
# batch_size = 100
# lr = 2e-4
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
y = train_data['y']
# X = X/255.
X = X.transpose([3, 0, 1, 2])
# X = np.reshape(X,(73257,3072))
# X = np.array(X)/float(255.)
pseudo_y = np.load('gist-0.364.npy')

# Y = train_data['y']
# data_pool = utils.MemoryData({'img': X, 'label':gist_feature}, batch_size)

#==============Resize_image=================
# real = tf.placeholder(tf.float32, shape=[None, 32,32,3])
# resize_op = tf.image.resize_images(real,size=[150,150])
# sess = tf.Session()
# resized_X = sess.run(resize_op, feed_dict={real: X[:1000]})
# sess.close()
#=================KERAS=====================
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, BatchNormalization,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

y_train = keras.utils.to_categorical(pseudo_y, 10)
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model.fit(X, y_train, batch_size=128, nb_epoch=15, verbose=1)

predict = model.predict(X)
y_classes = predict.argmax(axis=-1)
acc = my_utils.cluster_acc(y_classes, y)
# np.save('gist-0.364', all_y)
print('full-acc', acc[0])

