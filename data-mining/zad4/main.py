import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import NMF, PCA

IMAGES_DIR = './images'


def nmf(x, y, small_error: bool = False):
    n, m = x.shape
    unique_classes = np.unique(y)
    unique_classes_count = len(unique_classes)

    w = np.random.rand(n, 4)
    h = np.random.rand(4, m)

    max_iter = 1000
    e = 1.0e-10 if small_error is True else 0.0

    for n in range(max_iter):
        w_tx = w.T @ x
        w_twh = w.T @ w @ h + e

        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                h[i, j] = h[i, j] * w_tx[i, j] / w_twh[i, j]

        xh_t = x @ h.T
        whh_t = w @ h @ h.T + e

        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j] = w[i, j] * xh_t[i, j] / whh_t[i, j]

    return w, h


def experiment_nmf(h, w, build_in: bool = False, dataset_name: str = 'iris'):
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

        plt.title(f'{dataset_name} {membership_title} NMF')
        plt.savefig(f'{IMAGES_DIR}/nmf_{dataset_name}_{membership_title}{suffix_3d}.png')
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

        plt.title(f'PCA for {membership_title} NMF, {dataset_name}')
        plt.savefig(f'{IMAGES_DIR}/pca_{dataset_name}_{membership_title}_nmf{suffix_3d}.png')
        # plt.show()


def compare_experiment(h_own, h_build_in, y, dataset_name: str):
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

        plt.savefig(f'{IMAGES_DIR}/comparison_{dataset_name}_{"2d" if i == 0 else "3d"}.png')
        # plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    faces = datasets.fetch_olivetti_faces()

    for index, [dataset, dataset_name] in enumerate(zip([iris, faces], ['iris', 'faces'])):
        X = dataset.data
        Y = dataset.target

        h_own, w_own = nmf(X, Y, index > 0)
        experiment_nmf(h_own, w_own, dataset_name=dataset_name)

        model = NMF(max_iter=1000, n_components=4)
        h_build_in = model.fit_transform(X)
        w_build_in = model.components_
        experiment_nmf(h_build_in, w_build_in, build_in=True, dataset_name=dataset_name)

        compare_experiment(h_own, h_build_in, Y, dataset_name)

"""
próbka jest ułożona w kolumnie, a nie w wierszu
w pliku będzie oznaczane jako macierz V
W - ciekawe wzorce dla naszych irysów
H - współczynniki mówiące jak te wzorce wykorzystywać aby odtworzyć konkretną próbkę

staramy się uzyskać idealną reprezentację, czyli te macierze powinny przestać się zmieniać
w przypadku isysów należy założyć, że nie ma żadnych zmian
dla twarzy można sobie założyć mały błąd, bo inaczej może się to zakończyć niepowodzeniem

"""