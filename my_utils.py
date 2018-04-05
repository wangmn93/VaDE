import data_mnist as data
import utils
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def getMNISTDatapool(batch_size, keep=None, shift=True):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('MNIST_data',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('MNIST_data', keep=keep, shift=shift)
    print "Total number of training data: " + str(num_train_data)
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs, 'label':y}, batch_size)
    return data_pool

def getFashion_MNISTDatapool(batch_size, keep=None, shift=True):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST', keep=keep, shift=shift)
    print "Total number of training data: " + str(num_train_data)
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs, 'label':y}, batch_size)
    return data_pool

def getOnehot(labels, depth):
    onehot_labels = tf.one_hot(indices=labels, depth=depth)
    return onehot_labels

def getToyDataset(mus, cov, numberPerCluster):
    X = np.array([mus[0]])
    for mu in mus:
        x = np.random.multivariate_normal(mu, cov, numberPerCluster)
        # plt.plot(x[:, 0], x[:, 1], 'rx')
        # plt.plot(x2, y2, 'bx')

        X = np.concatenate((X,x))
    # plt.axis('equal')
    # plt.show()
    return X[1:]

def getToyDatapool(batch_size, mus, cov, numberPerCluster):
    X = getToyDataset(mus, cov, numberPerCluster)
    print "Total number of training data: " + str(len(X))
    data_pool = utils.MemoryData({'point':X}, batch_size)
    return data_pool

def saveSampleImgs(imgs, full_path, row, column):
    utils.imwrite(utils.immerge(imgs, row, column),full_path)


def sample_and_save(sess, list_of_generators, list_of_names, feed_dict, save_dir, rows = 10, columns = 10, normalize=False):
    # list_of_generators = [images_form_g1, images_form_g2]  # used for sampling images
    # list_of_names = ['g1-it%d.jpg' % total_it, 'g2-it%d.jpg' % total_it]
    # rows = 10
    # columns = 10
    # label_zero = np.zeros((rows * columns, 1))
    # label_one = np.ones((rows * columns, 1))
    sample_imgs = sess.run(list_of_generators, feed_dict=feed_dict)
    if normalize:
        for i in range(len(sample_imgs)):
            sample_imgs[i] = sample_imgs[i]*2.-1.
    # save_dir = dir + "/sample_imgs"
    utils.mkdir(save_dir + '/')
    for imgs, name in zip(sample_imgs, list_of_names):
        saveSampleImgs(imgs=imgs, full_path=save_dir + "/" + name, row=rows, column=columns)

# def sample_once(it, feed, sess):
#     rows = 10
#     columns = 10
#     feed = {random_z: np.random.normal(size=[rows * columns, z_dim])}
#     # list_of_generators = image_sets  # used for sampling images
#     # list_of_names = ['it%d-c%d.jpg' %(it,i) for i in range(len(image_sets))]
#     save_dir = dir + "/sample_imgs"
#     my_utils.sample_and_save(sess=sess, list_of_generators=list_of_generators, feed_dict=feed,
#                              list_of_names=list_of_names, save_dir=save_dir)
