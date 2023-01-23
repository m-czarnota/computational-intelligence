import numpy as np
from abc import ABC, abstractmethod


class AbstractDescriptor(ABC):
    def __init__(self):
        self.points_count: int = ...

    @abstractmethod
    def descript_image(self, image: np.array):
        ...

    @abstractmethod
    def _calc_centroid(self, points: np.array):
        ...

    @staticmethod
    def _find_contour_points(image: np.array):
        return np.array(list(zip(*np.where(image == 0))))

    def _select_points_from_array(self, array: np.array) -> np.array:
        indexes_distribution = np.linspace(0, array.shape[0] - 1, self.points_count).astype(int)
        distances = [array[0]]

        if indexes_distribution[1] - indexes_distribution[0] <= 1:
            return array

        index_iter = 1

        for index in indexes_distribution[1:]:
            prev_index = indexes_distribution[index_iter - 1]
            between_values = array[prev_index + 1:index]

            mean_value = np.mean(between_values)

            distances[index_iter - 1] = mean_value
            distances.append(mean_value)

            index_iter += 1

        return np.array(distances)
