from scipy import sparse
import math


def freq(x, prob: bool = True) -> list:
    if type(x) == sparse.csr_matrix or type(x) == sparse.csc_matrix:
        nonzero = x.nonzero()[0]
        uniques = set(nonzero)

        count_nonzero = len(nonzero)
        counts = {
            0: x.shape[0] - count_nonzero,
            1: count_nonzero
        }
        total = sum(counts.values())

        return [uniques, counts if prob is False else {key: val / total for key, val in counts.items()}]

    counts = {}
    uniques = []

    for val in x:
        if val not in uniques:
            uniques.append(val)

        if val in counts.keys():
            counts[val] += 1
            continue

        counts[val] = 1

    total = sum(counts.values())
    return [uniques, counts if prob is False else {key: val / total for key, val in counts.items()}]


def freq2(x, y, prob: bool = True) -> list:
    if (type(x) == sparse.csr_matrix or type(x) == sparse.csc_matrix) and (type(y) == sparse.csr_matrix or type(y) == sparse.csc_matrix):
        x_nonzero = x.nonzero()[0]
        y_nonzero = y.nonzero()[0]

        uniques_x = set(x_nonzero)
        uniques_y = set(y_nonzero)
        intersection_x_y = uniques_x.intersection(uniques_y)

        count_intersection = len(intersection_x_y)
        count_x_nonzero = len(x_nonzero)
        count_y_nonzero = len(y_nonzero)
        count_shared_zeros = x.shape[0] - count_x_nonzero + y.shape[0] - count_y_nonzero

        counts = {
            (0, 0): count_shared_zeros,
            (0, 1): count_y_nonzero - count_intersection,
            (1, 0): count_x_nonzero - count_intersection,
            (1, 1): count_intersection
        }
        # print(counts)
        total = sum(counts.values())

        return [uniques_x, uniques_y, counts if prob is False else {key: val / total for key, val in counts.items()}]

    counts = {}
    uniques = {'x': [], 'y': []}

    for x_val in x:
        if x_val not in uniques['x']:
            uniques['x'].append(x_val)

        for y_val in y:
            key = (x_val, y_val)

            if key not in counts.keys():
                counts[key] = 1

                if y_val not in uniques['y']:
                    uniques['y'].append(y_val)
            else:
                counts[key] += 1

    total = sum(counts.values())
    return [uniques['x'], uniques['y'], counts if prob is False else {key: val / total for key, val in counts.items()}]


def entropy(x, y=None, conditional_reverse: bool = False):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)

        if conditional_reverse is True and y is not None:
            uniques_x, probs_x = freq(x)
            entropy_y = entropy(y)

            return sum(prob * entropy_y for prob in probs_x.values())

    return -sum(prob * math.log2(prob) if prob != 0 else 0 for prob in probs.values())


def infogain(x, y, reverse: bool = False):
    if reverse is False:
        return entropy(x) + entropy(y) - entropy(x, y)
    return entropy(y) - entropy(x, y, conditional_reverse=True)


def kappa(x, y):
    return infogain(x, y) / entropy(y)


def gini(x, y=None, conditional_reverse: bool = False):
    if y is None:
        uniques, probs = freq(x)
    else:
        uniques_x, uniques_y, probs = freq2(x, y)

    if conditional_reverse is True and y is not None:
        uniques, probs = freq(x)
        gini_y = gini(y)
        return sum(prob * gini_y for prob in probs.values())

    return 1 - sum(prob ** 2 for prob in probs.values())


def ginigain(x, y):
    return gini(y) - gini(x, y, True)
