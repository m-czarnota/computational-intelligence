from LinearClassifier import LinearClassifier


class Perceptron(LinearClassifier):
    def fit(self, x, d):
        w, b = 0, 0
        n = 0

        while n < x.shape[0]:
            for i in range(x.shape[0]):
                print(d[i])
                print(d[i] * (x[i, :].dot(w) + b))

                if d[i] * (x[i, :].dot(w) + b) <= 0:
                    w += d[i] * x[i]
                    b += d[i]
                    n = 0
                else:
                    n += 1

        return w, b

