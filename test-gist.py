import gist
import scipy.io as sio
import numpy as np
from pyGist import gist as g2
train_data = sio.loadmat('../train_32x32.mat')

# access to the dict
X = train_data['X'][:10000]
Y = train_data['y']
X = X
X = X.transpose([3, 0, 1, 2])
print(X.shape)
gist_features = []
for counter, x in enumerate(X):
    descriptor = gist.extract(x)
    r = x[:,:,0]
    g = x[:,:,1]
    b = x[:,:,2]
    d2 = np.concatenate((g2(r),g2(g),g2(b)))
    gist_features.append(d2)
    print counter
gist_features = np.array(gist_features)
np.save('svhn-gist-2', gist_features)
print('finish extraction')
# from sklearn.cluster import KMeans
#
#             # imgs = full_data_pool.batch('img')
#             # imgs = (imgs + 1) / 2.
#
#             # sample = sess.run(z_mean, feed_dict={real: X})
# kmean = KMeans(n_clusters=10, n_init=20).fit(gist_features)
#             # centroids = kmean.cluster_centers_
# predict_y = kmean.predict(gist_features)
#             # predict_y = sess.run(predicts, feed_dict={real: X})
# import my_utils
# acc = my_utils.cluster_acc(predict_y, Y)
# print( acc[0])
# a = 0