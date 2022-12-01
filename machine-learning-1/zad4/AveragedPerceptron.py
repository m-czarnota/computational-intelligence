import time
import numpy as np

from Perceptron import Perceptron

"""
averaged uśredniamy sumę wag, dla tych która była dobra
uśredniamy wagi, lista w, b. ile razy wagi były dobre i uśredniamy

do margin dodajemy distance = True. jak jest 
jeset prosta. to odległość pkt od prostej to d=<w, x> + b / ||w|| -> odległość
margines (<w, x> + b) * di**2
jak distance == True to dzielimy przez ||w||, jak False to nie dzielimy

trzeba pomnożyć kolumna * kolumna, więc element przez element
w matlabie jest mnożenie .*

------------------
można robić wewnątrz pętli
średnia = suma(xi) / n
srednia(x[i+1]) = (x * i + x[i + 1]) / i + 1
zmienić predycję!! prezentacja

"""


class AveragedPerceptron(Perceptron):
    def __init__(self, max_seconds: int = 5):
        super().__init__(max_seconds=max_seconds)

        self.c_ = None

    def fit(self, x, d):
        self.class_labels_ = np.unique(d)

        w = averaged_w = np.ones(x.shape[1])
        b = n = beta = 0
        c = 1
        t1 = time.time()

        while n < x.shape[0]:
            t2 = time.time() - t1
            if t2 > self.max_seconds:
                print(f'Max time reached out: {t2}s! Break algorithm')
                break

            for i in np.arange(x.shape[0]):
                if d[i] * (x[i, :].dot(w) + b) > 0:
                    n += 1
                    c += 1
                    continue

                actual_w = d[i] * x[i]

                w += actual_w
                b += d[i]
                n = 0

                averaged_w += actual_w * c  # accumulate the curr weight values (good weights reused)
                beta += d[i] * c

                self.iteration_count += 1

        self.coefs_ = averaged_w
        self.intercepts_ = b
        self.c_ = c

        return self.coefs_, self.intercepts_
