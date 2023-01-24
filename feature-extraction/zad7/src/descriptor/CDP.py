from __future__ import annotations
import numpy as np

from src.descriptor.AbstractDescriptor import AbstractDescriptor


class CDP(AbstractDescriptor):
    def __init__(self, points_count: int = 200):
        super().__init__()

        self.points_count: int = points_count

        self.centroid = None
        self.distances = None

    def descript_image(self, image: np.array) -> CDP:
        points = self._find_contour_points(image)
        self.centroid = self._calc_centroid(points)
        self.distances = self.__calc_distances_to_centroid(points)
        self.distances = self._select_points_from_array(self.distances)

        return self

    def calc_distance_to_other_descriptor(self, descriptor: CDP) -> float:
        distances = self.distances - descriptor.distances
        distances = np.sqrt(distances ** 2)

        return np.mean(distances)

    def _calc_centroid(self, points: np.array) -> np.array:
        y_sum = np.sum(points[:, 0])
        x_sum = np.sum(points[:, 1])

        return np.array([x_sum / points.size, y_sum / points.size])

    def __calc_distances_to_centroid(self, points: np.array) -> np.array:
        distances = np.empty(points.shape[0])

        for point_iter, point in enumerate(points):
            distances[point_iter] = self.__calc_distance_to_centroid(point)

        return distances

    def __calc_distance_to_centroid(self, point: np.array) -> float:
        return np.sqrt((point[1] - self.centroid[1]) ** 2 + (point[0] - self.centroid[0]) ** 2)
