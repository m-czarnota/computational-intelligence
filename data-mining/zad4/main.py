import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import NMF, PCA

IMAGES_DIR = './images'


def nmf(x, y):
    n, m = x.shape
    unique_classes = np.unique(y)
    unique_classes_count = len(unique_classes)

    w = np.random.rand(n, unique_classes_count)
    h = np.random.rand(unique_classes_count, m)

    max_iter = 1000

    for n in range(max_iter):
        w_tx = w.T @ x
        w_twh = w.T @ w @ h

        for i in range(np.size(h, 0)):
            for j in range(np.size(h, 1)):
                h[i, j] = h[i, j] * w_tx[i, j] / w_twh[i, j]

        xh_t = x @ h.T
        whh_t = w @ h @ h.T

        for i in range(np.size(w, 0)):
            for j in range(np.size(w, 1)):
                w[i, j] = w[i, j] * xh_t[i, j] / whh_t[i, j]

    return w, h


def experiment_nmf(h, w, build_in: bool = False):
    membership_title = 'build_in' if build_in is True else 'own'

    for i in range(2):
        suffix_3d = "_3d" if i != 0 else ""

        fig = plt.figure(figsize=(20, 10))

        if i == 0:
            ax = fig.add_subplot()
            ax.scatter(h[:, 0], h[:, 1], c=Y)
        else:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(h[:, 0], h[:, 1], h[:, 2], c=Y)

        plt.title(f'{membership_title} NMF')
        plt.savefig(f'{IMAGES_DIR}/nmf_{membership_title}{suffix_3d}.png')
        # plt.show()

    pca = PCA()
    pca.fit(h)
    h_pca = pca.transform(h)

    for i in range(2):
        suffix_3d = "_3d" if i != 0 else ""

        fig = plt.figure(figsize=(20, 10))

        if i == 0:
            ax = fig.add_subplot()
            ax.scatter(h_pca[:, 0], h_pca[:, 1], c=Y)
        else:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(h_pca[:, 0], h_pca[:, 1], h_pca[:, 2], c=Y)

        plt.title(f'PCA for {membership_title} NMF')
        plt.savefig(f'{IMAGES_DIR}/pca_{membership_title}_nmf{suffix_3d}.png')
        # plt.show()


def compare_experiment(h_own, h_build_in, y):
    pca = PCA()
    pca.fit(h_build_in)
    h_pca = pca.transform(h_build_in)

    for i in range(2):
        fig = plt.figure(figsize=(20, 10))

        for index, [h, title] in enumerate(zip([h_own, h_build_in, h_pca], ['own NMF', 'build-in NMF', 'PCA NMF'])):
            if i != 0:
                ax = fig.add_subplot(1, 3, index + 1, projection='3d')
                ax.scatter(h[:, 0], h[:, 1], h[:, 2], c=y)
            else:
                ax = fig.add_subplot(1, 3, index + 1)
                ax.scatter(h[:, 0], h[:, 1], c=y)

            ax.set_title(title)

        plt.savefig(f'{IMAGES_DIR}/comparison_{"2d" if i == 0 else "3d"}.png')
        # plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    h_own, w_own = nmf(X, Y)
    # print(w, h)
    experiment_nmf(h_own, w_own)

    model = NMF(max_iter=1000)
    h_build_in = model.fit_transform(X)
    w_build_in = model.components_
    # print(w, h)
    experiment_nmf(h_build_in, w_build_in, build_in=True)

    compare_experiment(h_own, h_build_in, Y)

"""
próbka jest ułożona w kolumnie, a nie w wierszu
w pliku będzie oznaczane jako macierz V
W - ciekawe wzorce dla naszych irysów
H - współczynniki mówiące jak te wzorce wykorzystywać aby odtworzyć konkretną próbkę

staramy się uzyskać idealną reprezentację, czyli te macierze powinny przestać się zmieniać
w przypadku isysów należy założyć, że nie ma żadnych zmian
dla twarzy można sobie założyć mały błąd, bo inaczej może się to zakończyć niepowodzeniem

"""