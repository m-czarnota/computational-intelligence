from typing import Tuple
import numpy as np
import cv2

from src.descriptor.AbstractDescriptor import AbstractDescriptor


class LogPol(AbstractDescriptor):
    """
    Converts the original image (x, y) into another (p, w) in which the angular coordinates are placed on the vertical
    axis and the logarithm of the radius coordinates are placed on the horizontal one (furthermore a normalization has
    to be carried out in order to implement the transformation).
    """
    def __init__(self):
        super().__init__()

        self._folder_name = 'log_pol'

        self.centroid = None
        self.p = None  # logarytm współrzędnych promienia
        self.w = None  # współrzędne kątowe

    def descript_image(self, image: np.array):
        points = self._find_contour_points(image)
        self.centroid = self._calc_centroid(points)
        self.p, self.w = self.__calc_distances_to_centroid(points)

    def save_image(self, main_folder: str, filename: str):
        super().save_image(main_folder, filename)
        cv2.imwrite(f'{self._folder_path}/{filename}', self.distances)

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
