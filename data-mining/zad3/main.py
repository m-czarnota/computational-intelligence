import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

IMAGES_DIR = './images'


def find_perm(y_real, y_predict):
    conf_matrix = confusion_matrix(y_real, y_predict)
    conf_matrix_argmax = conf_matrix.argmax(axis=0)

    return np.array([conf_matrix_argmax[i] for i in y_predict])


def visualize2d(x_reduced, classes_set: list, plot_title: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_size_inches(20, 10)

    for ax, y, title in zip([ax1, ax2], classes_set, ['real classes', 'algorithm classes']):
        ax.set_title(title)
        ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y)

    y, permutations = classes_set
    ax3.set_title('data classified correctly and not correctly')

    scatter = ax3.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y == permutations)
    ax3.legend(*scatter.legend_elements())

    plt.savefig(f'{IMAGES_DIR}/{plot_title}_2D.png')
    # plt.show()


def visualize3d(x_reduced, classes_set: list, plot_title: str):
    fig = plt.figure(figsize=(20, 10))

    for index, [y, title] in enumerate(zip(classes_set, ['real classes', 'algorithm classes'])):
        ax = fig.add_subplot(1, 3, index + 1, projection='3d')
        ax.set_title(title)
        ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=y)

    y, permutations = classes_set
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('data classified correctly and not correctly')

    scatter = ax3.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=y == permutations)
    ax3.legend(*scatter.legend_elements())

    plt.savefig(f'{IMAGES_DIR}/{plot_title}_3D.png')
    # plt.show()


def iris_experiment():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    k_means = KMeans()
    predict = k_means.fit_predict(x)

    perms = find_perm(y, predict)
    # print(perms)

    jac_score = jaccard_score(y, perms, average=None)
    print('iris jaccard score:', jac_score)

    for i, method in zip(range(2, 4), [visualize2d, visualize3d]):
        pca = PCA(n_components=i)
        x_reduced = pca.fit_transform(x)

        method(x_reduced, [y, perms], 'iris_pca')


if __name__ == '__main__':
    iris_experiment()

    """
    Jaccarda mierzy podobienstwo mirdzy dwoma zbiorami i jest zdefiniowany jako iloraz mocy czesci wspolnej zbiorow i mocy sumy tych zbiorow.
    Wartosci przyjmowane przez wspolczynnik Jaccarda zawieraja sie w podzbiorze zbioru liczb rzeczywistych <0,1>. 
    Jeśli wspolczynnik Jaccarda przyjmuje wartosci bliskie zeru, zbiory są od siebie róozne, natomiast gdy jest bliski 1, zbiory są do siebie podobne.
    """
