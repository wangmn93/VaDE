import parametric_tSNE
import my_utils
from matplotlib import pyplot as plt
from tensorflow.contrib.keras import layers

all_layers = [layers.Dense(500, input_shape=(784,), activation='relu', kernel_initializer='glorot_uniform'),
layers.Dense(500, activation='relu', kernel_initializer='glorot_uniform'),
layers.Dense(2000, activation='relu', kernel_initializer='glorot_uniform'),
layers.Dense(10, activation='linear', kernel_initializer='glorot_uniform')]

test_data_list, numPerClass = my_utils.getTest_data(numPerClass=100, reshape=False)
colors =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown']
#             0       1       2       3        4          5        6        7         8       9


X, Y = my_utils.load_data('mnist')
high_dims = 784
num_outputs = 10
perplexity = 30
num_data = 20000
ptSNE = parametric_tSNE.Parametric_tSNE(high_dims, num_outputs, perplexity,do_pretrain=False,all_layers=all_layers)

# ptSNE.fit(X[:num_data], epochs=100, verbose=1)
def test_acc(X):
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    output_res = ptSNE.transform(X)
    predict_y = KMeans(n_clusters=10, n_init=20).fit_predict(output_res)
    acc = my_utils.cluster_acc(predict_y, Y[:len(X)])
    print('full-acc',acc[0])

    # plt.clf()
    # sample = ptSNE.transform(test_data_list)
    # X_embedded = TSNE(n_components=2).fit_transform(sample)
    # for i in range(10):
    #     plt.scatter(X_embedded[i * 100:(i + 1) * 100, 0], X_embedded[i * 100:(i + 1) * 100, 1], color=colors[i],
    #                 label=str(i), s=2)
    #     # for test_d in test_data:
    #     #     sample = sess.run(z_mean, feed_dict={real: test_d})
    #     #     # X_embedded = sample
    #     #     X_embedded = TSNE(n_components=2).fit_transform(sample)
    #     #     plt.scatter(X_embedded[:,0],X_embedded[:,1],color=colors[i],label=str(i), s=2)
    #     #     i += 1
    #     # plt.draw()
    # plt.legend(loc='best')
    # plt.show()

ptSNE.pretrain(X[:num_data], epochs=50, verbose=1)
# test_acc(X)
ptSNE.save_model("./test_model.h5")
# ptSNE.restore_model("./test_model.h5", num_perplexities=perplexity)
test_acc(X[:num_data])
# for _ in range(10):
ptSNE.finetune(X[:num_data], epochs=10, verbose=1)
    # ptSNE.pretrain(X[:num_data], epochs=1, verbose=1)
# ptSNE.restore_model("./test_model.h5")
# ptSNE.fit(X[:num_data], epochs=10, verbose=1)
test_acc(X[:num_data])

ptSNE.pretrain(X[:num_data], epochs=100, verbose=1)
test_acc(X[:num_data])
# ptSNE.save_model("./test_model-finetune.h5")

ptSNE.finetune(X[:num_data], epochs=10, verbose=1)
test_acc(X[:num_data])

ptSNE.pretrain(X[:num_data], epochs=100, verbose=1)
test_acc(X[:num_data])