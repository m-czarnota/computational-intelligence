import cv2
import numpy as np

from src.descriptor.AbstractDescriptor import AbstractDescriptor


class CDP(AbstractDescriptor):
    def __init__(self):
        super().__init__()

        self._folder_name = 'cdp'

        self.centroid = None
        self.distances = None

    def descript_image(self, image: np.array):
        points = self._find_contour_points(image)
        self.centroid = self._calc_centroid(points)
        self.distances = self.__calc_distances_to_centroid(points)

    def save_image(self, main_folder: str, filename: str):
        super().save_image(main_folder, filename)
        cv2.imwrite(f'{self._folder_path}/{filename}', self.distances)  # ?? no 1 vector, we need a matrix

    def _calc_centroid(self, points: np.array):
        y_sum = np.sum(points[:, 0])
        x_sum = np.sum(points[:, 1])

        return np.array([x_sum / points.size, y_sum / points.size])

    def __calc_distances_to_centroid(self, points: np.array):
        distances = np.empty(points.shape[0])

        for point_iter, point in enumerate(points):
            distances[point_iter] = self.__calc_distance_to_centroid(point)

        return distances

    def __calc_distance_to_centroid(self, point: np.array):
        return np.sqrt((point[1] - self.centroid[1]) ** 2 + (point[0] - self.centroid[0]) ** 2)
