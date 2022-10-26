import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, jaccard_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
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
    plt.close(fig)


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
    plt.close(fig)


def visualise_dendrogram(y, y_predict, method_name: str):
    joined_y = np.c_[(y, y_predict)]
    z = hierarchy.linkage(joined_y, method='average')

    fig = plt.figure(figsize=(20, 10))

    hierarchy.dendrogram(z)
    plt.title(f'dendrogram_{method_name}')

    plt.savefig(f'{IMAGES_DIR}/dendrogram_{method_name}.png')
    # plt.show()
    plt.close(fig)


def clusterize(x, y, method, plot_dendrogram: bool = False, dataset_name: str = 'iris'):
    method_name = method.__str__()

    predict = method.fit_predict(x)
    perms = find_perm(y, predict)
    # print(perms)

    jac_score = jaccard_score(y, perms, average=None)
    print(f'{dataset_name} jaccard score {method_name}:', jac_score)

    for i, visualization in zip(range(2, 4), [visualize2d, visualize3d]):
        pca = PCA(n_components=i)
        x_reduced = pca.fit_transform(x)

        visualization(x_reduced, [y, perms], f'{dataset_name}_{method_name}')

    if plot_dendrogram is True:
        visualise_dendrogram(y, predict, f'{dataset_name}_{method_name}')


def experiment(x, y, dataset_name: str = 'iris'):
    k_means = KMeans()
    clusterize(x, y, k_means, dataset_name=dataset_name)

    gmm = GaussianMixture()
    clusterize(x, y, gmm, dataset_name=dataset_name)

    linkages = ['ward', 'complete', 'average', 'single']
    for linkage in linkages:
        ag = AgglomerativeClustering(linkage=linkage)
        clusterize(x, y, ag, True, dataset_name=dataset_name)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    experiment(X, Y)

    zoo = pd.read_csv('zoo.csv')

    Y = zoo['type'].values
    Y_uniques = np.unique(Y)
    Y_mapped = [np.where(Y_uniques == animal_type)[0][0] for animal_type in Y]

    X = zoo.drop(['animal', 'type'], axis=1)

    experiment(X, Y_mapped, 'zoo')

    """
    Co to jest indeks Jaccarda i jaka jest jego interpretacja?
    Jaccarda mierzy podobienstwo mirdzy dwoma zbiorami i jest zdefiniowany jako iloraz mocy czesci wspolnej zbiorow i mocy sumy tych zbiorow.
    Wartosci przyjmowane przez wspolczynnik Jaccarda zawieraja sie w podzbiorze zbioru liczb rzeczywistych <0,1>. 
    Jeśli wspolczynnik Jaccarda przyjmuje wartosci bliskie zeru, zbiory są od siebie róozne, natomiast gdy jest bliski 1, zbiory są do siebie podobne.
    
    W jaki sposób działają metody aglomeracyjne?
    * metoda najblizszego sasiedztwa: minimalne odleglosci miedzy wszystkimi obserwacjami dwoch zbiorow
    * metoda srednich polaczen: srednia odleglosc kazdej obserwacji dwoch zbiorow
    * metoda najdalszych polaczen: maksymalne odleglosci miedzy wszystkimi obserwacjami dwoch zbiorow
    * metoda warda: minimalizuje wariacnje laczenia klastrow
    """
