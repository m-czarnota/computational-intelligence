from __future__ import annotations
import numpy as np

from src.descriptor.AbstractDescriptor import AbstractDescriptor


class CentroidPDH(AbstractDescriptor):
    def __init__(self, points_count: int = 5):
        super().__init__()

        self.points_count: int = points_count  # r in PDH terminology

        self.centroid = None
        self.h = None

    def descript_image(self, image: np.array) -> CentroidPDH:
        points = self._find_contour_points(image)
        self.centroid = self._calc_centroid(points)

        oi = self.__calc_o(points)
        pi = self.__calc_p(points)

        for oi_iter, oi_val in enumerate(oi):
            oi[oi_iter] = np.floor(oi_val) if oi_val - np.floor(oi_val) < 0.5 else np.ceil(oi_val)

        oi, pi = self.__sort_oi_and_pi_by_oi(oi, pi)

        pk = self.__calc_pk_from_sorted_oi_and_pi(oi, pi)
        pk = self.__normalize_vector(pk)

        lk = self.__calc_lk(pk)
        lk = self.__normalize_vector(lk)

        self.h = self.__calc_h(lk)

        return self

    def calc_distance_to_other_descriptor(self, descriptor: CentroidPDH) -> float:
        if self.h.size < descriptor.h.size:
            return np.mean(self.h)

        distances = self.h[:descriptor.h.shape[0]] - descriptor.h
        distances = np.sqrt(distances ** 2)

        return np.mean(distances)

    def _calc_centroid(self, points: np.array):
        y_sum = np.sum(points[:, 0])
        x_sum = np.sum(points[:, 1])

        return np.array([x_sum / points.size, y_sum / points.size])

    def __calc_o(self, points: np.array):
        o = np.empty(points.shape[0])

        for point_iter, point in enumerate(points):
            o[point_iter] = np.arctan((point[0] - self.centroid[0]) / (point[1] - self.centroid[1]))

        return o

    def __calc_p(self, points: np.array):
        points = points.copy().astype('float')

        for centroid_iter, centroid_val in enumerate(self.centroid):
            points[:, centroid_iter] -= centroid_val
            points[:, centroid_iter] **= 2

        return points.sum(axis=1)

    @staticmethod
    def __sort_oi_and_pi_by_oi(oi: np.array, pi: np.array):
        return zip(*sorted(zip(oi, pi)))

    @staticmethod
    def __calc_pk_from_sorted_oi_and_pi(oi: np.array, pi: np.array):
        uniques_oi = np.unique(oi)

        buckets = np.empty(uniques_oi.size, dtype=object)
        buckets[...] = [[] for _ in range(buckets.shape[0])]

        for oi_iter, oi_val in enumerate(oi):
            bucket_number = np.where(uniques_oi == oi_val)[0][0]
            pi_val = pi[oi_iter]

            buckets[bucket_number].append(pi_val)

        return np.array(list(map(lambda pi_list: np.max(pi_list), buckets)))

    @staticmethod
    def __normalize_vector(vector: np.array):
        max_val = np.max(vector)

        return vector / max_val

    def __calc_lk(self, pk: np.array):
        lk = np.empty(pk.size)

        for pk_iter, pk_val in enumerate(pk):
            lk[pk_iter] = self.points_count if pk_val == 1 else np.floor(self.points_count * pk_val)

        return lk

    @staticmethod
    def __calc_h(lk: np.array):
        h = np.empty(lk.size)

        for lk_iter, lk_val in enumerate(lk):
            h[lk_iter] = 1 if lk_iter == lk_val else 0

        return h
