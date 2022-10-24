import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import stats


def find_perm(Y_real, Y_pred):
    """
    funkcja inicjalizuje pusty wektor, ktory bedzie przechowywal permutacje, ktore: wartosci z Y_pred najbardziej odpowiadają tym z Y_real
    dla wszystkich klastrow:
        idx to indeksy Y_pred, gdzie ich wartosci pod tymi indeksami sa rowne i
        zapamietanie pierwszej wartosci modalnej/dominanty znajdujacej sie w Y_real pod kolumnami o indeksach zawartymi w idx
        zapamietanie powyzszej wartosci
    zwrocenie tych permutacji, ktore znajduja sie w Y_pred
    """
    clusters = len(np.unique(Y_real))
    perm = []
    for i in range(clusters):
        idx = Y_pred == i
        new_label = stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    k_means = KMeans()
    k_means.fit(X_train, y_train)
    print(Y)
    print(y_train)
    print(k_means.labels_)

    array = np.array([0, 0, 0, 2, 2, 2, 3, 5, 7, 2, 2, 0, 0, 1, 1, 1])
    print(array[array == 2])

    perms = find_perm(y_train, k_means.labels_)
    # print(perms)
