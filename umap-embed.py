import umap
import my_utils
import numpy as np
import data_mnist as data
# from sklearn.datasets import load_digits
X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
X = np.reshape(X, [70000,784])
# digits = load_digits()

from sklearn.manifold import TSNE
embedding = TSNE(n_components=10).fit_transform(X[:5000])
# embedding = umap.UMAP(n_components=100,
#                       n_neighbors=5,
#                       min_dist=0.1,
#                       metric='correlation').fit_transform(X)
from sklearn.cluster import KMeans

            # imgs = full_data_pool.batch('img')
            # imgs = (imgs + 1) / 2.

            # sample = sess.run(z_mean, feed_dict={real: X})
kmean = KMeans(n_clusters=10, n_init=20).fit(embedding)
            # centroids = kmean.cluster_centers_
predict_y = kmean.predict(embedding)
            # predict_y = sess.run(predicts, feed_dict={real: X})
acc = my_utils.cluster_acc(predict_y, Y[:5000])
print('full-acc', acc[0])
            # i = 0