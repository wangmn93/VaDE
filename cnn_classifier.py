from __future__ import print_function

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# x_image = tf.reshape(x, [-1, 32, 32, 3])
def cnn_classifier(x, keep_prob,reuse=True, name='classifier' ):
    with tf.variable_scope(name, reuse=reuse):
    # Convolutional layer 1
      W_conv1 = weight_variable([5, 5, 3, 32])
      b_conv1 = bias_variable([32])

      h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)

    # Convolutional layer 2
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])

      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      print(h_pool2.shape)
    # Fully connected layer 1
      h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

      W_fc1 = weight_variable([8 * 8 * 64, 1024])
      b_fc1 = bias_variable([1024])

      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    # keep_prob  = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully connected layer 2 (Output layer)
      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])

      logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      return logits

if __name__ == '__main__':
      # Input layer
  x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
  k_prob = tf.placeholder(tf.float32, [], name='keep_prob')
  import models_mnist as models
  y_p = tf.nn.softmax(models.cnn_classifier2(x, reuse=False, keep_prob=k_prob))

  # Evaluation functions
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_p), reduction_indices=[1]))
  pred = tf.argmax(y_p, 1)
  correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

  # Training algorithm
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  # Training steps
  with tf.Session() as sess:
    import scipy.io as sio
    import numpy as np
    train_data = sio.loadmat('../train_32x32.mat')
    X = train_data['X'] / 127.5 - 1
    y = train_data['y'] - 1

    # Y = train_data['y']
    pseudo_y = np.load('svhn-peusdo-label-0.47.npy')
    pseudo_y_onehot = tf.keras.utils.to_categorical(
      pseudo_y,
      num_classes=10
    )
    X = X.transpose([3, 0, 1, 2])
    # data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
    import utils

    data_pool = utils.MemoryData({'img': X, 'label': pseudo_y_onehot}, 50)
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.initialize_all_variables())

    max_steps = 5000
    for step in range(max_steps):
      batch_xs, batch_ys = data_pool.batch()
      if (step % 100) == 0:
        print(step, sess.run(accuracy, feed_dict={x: X[:1000], y_: pseudo_y_onehot[:1000], k_prob: 1.0}))
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, k_prob: 0.5})
    print(max_steps, sess.run(accuracy, feed_dict={x: X[:5000], y_: pseudo_y_onehot[:5000], k_prob: 1.0}))
    # saver.restore(sess, 'results/cnn-classifier-model.ckpt')

    import my_utils
    predicts = sess.run(pred, feed_dict={x: X[:5000], k_prob: 1.0})
    acc = my_utils.cluster_acc(predicts, y[:5000])
    # np.save('gist-0.364', all_y)
    print('full-acc', acc[0])
    save_path = saver.save(sess, "results/cnn-classifier-model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(" [*] Close main session!")
    sess.close()