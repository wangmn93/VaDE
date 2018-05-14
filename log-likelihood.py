import my_utils
import numpy as np
X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
#X, Y = my_utils.load_data('mnist')
X = np.reshape(X, [70000,784])
x_1 = X[:10000]
x_2 = X[20000:30000]
# num_data = 70000
import time

start = time.time()


from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_1)
print kde.score(x_2)/len(x_2)
end = time.time()
print end - start