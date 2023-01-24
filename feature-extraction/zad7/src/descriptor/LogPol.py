from __future__ import annotations
from typing import Tuple
import numpy as np

from src.descriptor.AbstractDescriptor import AbstractDescriptor


class LogPol(AbstractDescriptor):
    """
    Converts the original image (x, y) into another (p, w) in which the angular coordinates are placed on the vertical
    axis and the logarithm of the radius coordinates are placed on the horizontal one (furthermore a normalization has
    to be carried out in order to implement the transformation).
    """
    def __init__(self, points_count: int = 200):
        super().__init__()

        self.points_count: int = points_count

        self.centroid = None
        self.p = None  # logarytm współrzędnych promienia
        self.w = None  # współrzędne kątowe

    def descript_image(self, image: np.array) -> LogPol:
        points = self._find_contour_points(image)
        self.centroid = self._calc_centroid(points)
        self.p, self.w = self.__calc_distances_to_centroid(points)

        self.p = self._select_points_from_array(self.p)
        self.w = self._select_points_from_array(self.w)

        return self

    def calc_distance_to_other_descriptor(self, descriptor: LogPol) -> float:
        distances_p = np.sqrt((self.p - descriptor.p) ** 2)
        distances_w = np.sqrt((self.p - descriptor.w) ** 2)

        return np.sqrt((np.mean(distances_p) - np.mean(distances_w)) ** 2)

    def _calc_centroid(self, points: np.array):
        y_sum = np.sum(points[:, 0])
        x_sum = np.sum(points[:, 1])

        return np.array([x_sum / points.size, y_sum / points.size])

    def __calc_distances_to_centroid(self, points: np.array) -> Tuple:
        p = np.empty(points.shape[0])
        w = np.empty(points.shape[0])

        for point_iter, point in enumerate(points):
            p[point_iter] = self.__calc_p_for_point(point)
            w[point_iter] = self.__calc_w_for_point(point)

        return p, w

    def __calc_p_for_point(self, point: np.array):
        return np.log(np.sqrt((point[1] - self.centroid[1]) ** 2 + (point[0] - self.centroid[0]) ** 2))

    def __calc_w_for_point(self, point: np.array):
        return np.arctan((point[0] - self.centroid[0]) / (point[1] - self.centroid[1]))
