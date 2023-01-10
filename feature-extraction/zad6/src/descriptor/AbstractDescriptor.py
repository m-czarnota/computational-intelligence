import numpy as np
import os
from abc import ABC, abstractmethod


class AbstractDescriptor(ABC):
    def __init__(self):
        self._folder_name: str = ...
        self._folder_path: str = ...

    @abstractmethod
    def descript_image(self, image: np.array):
        ...

    @abstractmethod
    def save_image(self, main_folder: str, filename: str):
        self._folder_path = f'{main_folder}/{self._folder_name}'
        os.makedirs(self._folder_path, exist_ok=True)

    @abstractmethod
    def _calc_centroid(self, points: np.array):
        ...

    @staticmethod
    def _find_contour_points(image: np.array):
        return np.array(list(zip(*np.where(image == 0))))
