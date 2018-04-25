import data_mnist as data
import utils
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio
import sys
import gzip
from six.moves import cPickle

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def load_data(dataset):
    path = 'dataset/' + dataset + '/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))

    if dataset == 'reuters10k':
        data = scio.loadmat(path + 'reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()

    if dataset == 'har':
        data = scio.loadmat(path + 'HAR.mat')
        X = data['X']
        X = X.astype('float32')
        Y = data['Y'] - 1
        X = X[:10200]
        Y = Y[:10200]

    return X, Y

def getTest_data(numPerClass=500, reshape=True):
    test_data = [[], [], [], [], [], [], [], [], [], []]
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
    # plt.ion()  # enables interactive mode
    X, Y = load_data('mnist')
    if reshape:
        X = np.reshape(X, [70000, 28, 28, 1])
    for i, j in zip(X, Y):
        if len(test_data[j]) < numPerClass:
            test_data[j].append(i)
    test_data_list = test_data[0]
    for i in range(1, 10):
        test_data_list = np.concatenate((test_data_list, test_data[i]))
    return test_data_list, numPerClass

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_full_cifar_10(shift):
    img = None
    label = None
    for i in range(1, 5):
        if i == 1:
            batch = unpickle('cifar-10-batches-py/data_batch_%d' % i)
            img = batch['data']
            label = batch['labels']
        else:
            batch = unpickle('cifar-10-batches-py/data_batch_%d' % i)
            img = np.concatenate((img, batch['data']))
            label = np.concatenate((label, batch['labels']))
    batch = unpickle('cifar-10-batches-py/test_batch')
    img = np.concatenate((img, batch['data']))
    label = np.concatenate((label, batch['labels']))
    if shift:
        return img / 127.5 - 1, label
    else:
        return img / 255., label

def get_FullCifar10Datapool(batch_size, shift):
    imgs, labels = load_full_cifar_10(shift)
    imgs = np.reshape(imgs, [len(imgs), 3, 32, 32])
    imgs = imgs.transpose([0, 2, 3, 1])
    data_pool = utils.MemoryData({'img': imgs, 'label': labels},
                                 batch_size)
    return data_pool

def getMNISTDatapool(batch_size, shift, keep=None):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('MNIST_data',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('MNIST_data', keep=keep, shift=shift)
    print "Total number of training data: " + str(num_train_data)
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs, 'label':y}, batch_size)
    return data_pool

def getFullMNISTDatapool(batch_size, shift, keep=None):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('MNIST_data',shift=shift)
        imgs_t, y_t, num_train_data_t = data.mnist_load('MNIST_data',dataset='test',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('MNIST_data', keep=keep, shift=shift)
        imgs_t, y_t, num_train_data_t = data.mnist_load('MNIST_data', keep=keep,dataset='test',shift = shift)
    print "Total number of training data: " + str(num_train_data + num_train_data_t)
    imgs.shape = imgs.shape + (1,)
    imgs_t.shape = imgs_t.shape + (1,)
    data_pool = utils.MemoryData({'img': np.concatenate((imgs,imgs_t)), 'label':np.concatenate((y,y_t))}, batch_size)
    return data_pool

def getFashion_MNISTDatapool(batch_size, shift, keep=None):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST', keep=keep, shift=shift)
    print "Total number of training data: " + str(num_train_data)
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs, 'label':y}, batch_size)
    return data_pool

def getFullFashion_MNISTDatapool(batch_size, shift,keep=None):
    if keep is None:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST',shift=shift)
        imgs_t, y_t, num_train_data_t = data.mnist_load('Fashion_MNIST',dataset='test',shift=shift)
    else:
        imgs, y, num_train_data = data.mnist_load('Fashion_MNIST', keep=keep, shift=shift)
        imgs_t, y_t, num_train_data_t = data.mnist_load('Fashion_MNIST', keep=keep,dataset='test',shift = shift)
    print "Total number of training data: " + str(num_train_data + num_train_data_t)
    imgs.shape = imgs.shape + (1,)
    imgs_t.shape = imgs_t.shape + (1,)
    data_pool = utils.MemoryData({'img': np.concatenate((imgs,imgs_t)), 'label':np.concatenate((y,y_t))}, batch_size)
    return data_pool

def loadFullFashion_MNSIT(shift):
    imgs, y, num_train_data = data.mnist_load('Fashion_MNIST', shift=shift)
    imgs_t, y_t, num_train_data_t = data.mnist_load('Fashion_MNIST', dataset='test', shift=shift)
    imgs.shape = imgs.shape + (1,)
    imgs_t.shape = imgs_t.shape + (1,)
    return np.concatenate((imgs, imgs_t)), np.concatenate((y, y_t))

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


def smorms3(cost, params, lr=np.float32(0.001), eps=np.float32(1e-16)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        m = tf.Variable(np.ones(param.get_shape(), dtype='float32'), name='m')
        g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g')
        g2 = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g2')
        r = 1./(m + 1.)
        updates.append(g.assign((1. - r)*g + r*g_param))
        updates.append(g2.assign((1. - r)*g2 + r*g_param**2))
        with tf.control_dependencies(updates):
            x = g**2/(g2 + eps)
            updates.append(param.assign(param - tf.minimum(lr, x)/tf.sqrt(g2 + eps)*g_param))
            updates.append(m.assign(1. + (1. - x)*m))
    return updates