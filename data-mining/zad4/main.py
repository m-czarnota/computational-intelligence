import numpy as np
from sklearn import datasets


def nmf(x, y):
    n, m = x.shape
    unique_classes = np.unique(y)
    unique_classes_count = len(unique_classes)

    w = np.ones((n, unique_classes_count))
    h = np.ones((unique_classes_count, m))

    print(w.shape, h.shape)

    n = 0
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i, j] = h[i, j] * ((np.transpose(w ** n) * x)[i, j] / (np.transpose(w ** n) * (w ** n) * (h ** n))[i, j])
            w[i, j] = w[i, j] * ((x * np.transpose(h))[i, j] / (w * h * np.transpose(h))[i, j])

    n += 1


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    nmf(X, Y)

"""
próbka jest ułożona w kolumnie, a nie w wierszu
w pliku będzie oznaczane jako macierz V
W - ciekawe wzorce dla naszych irysów
H - współczynniki mówiące jak te wzorce wykorzystywać aby odtworzyć konkretną próbkę

staramy się uzyskać idealną reprezentację, czyli te macierze powinny przestać się zmieniać
w przypadku isysów należy założyć, że nie ma żadnych zmian
dla twarzy można sobie założyć mały błąd, bo inaczej może się to zakończyć niepowodzeniem

"""