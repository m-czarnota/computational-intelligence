import cmath
import numpy as np

from functions import transform_data_to_db_scale


class Filter:
    def __init__(self, data: np.array, max_freq: int):
        self.data = data
        self.max_freq = max_freq
        self.fs = max_freq * 2

        self.amplitude_characteristic: np.array = None
        self.amplitude_characteristic_decibel: np.array = None
        self.phase_characteristic: np.array = None

        self.boundary_decibel: int = None
        self.boundary_frequencies: np.array = None
        self.bandwidth: int = None

    def get_amplitude_characteristic(self) -> np.array:
        if self.amplitude_characteristic is None:
            self.calc_freq_characteristic(True)

        return self.amplitude_characteristic

    def get_phase_characteristic(self) -> np.array:
        if self.phase_characteristic is None:
            self.calc_freq_characteristic(True)

        return self.phase_characteristic

    def get_amplitude_characteristic_decibel(self) -> np.array:
        if self.amplitude_characteristic_decibel is None:
            self.calc_freq_characteristic_params(True)

        return self.amplitude_characteristic_decibel

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def calc_freq_characteristic(self, force: bool = False) -> None:
        if force is False and self.amplitude_characteristic is not None and self.phase_characteristic is not None:
            return

        h_vals = np.empty(self.max_freq, dtype='object')
        angle = np.empty(self.max_freq, dtype='object')

        for freq_iter, freq in enumerate(range(self.max_freq)):
            omega = 2 * np.pi * freq / self.fs
            # h = 0
            h = np.sum(self.data) * np.exp(-1j * omega * np.arange(self.shape[0]))

            # for fir_iter, fir_val in enumerate(self.data):
            #     h += fir_val * np.exp(-1j * omega * fir_iter)

            h_vals[freq_iter] = h
            # angle[freq_iter] = np.degrees(np.angle(h))

        self.amplitude_characteristic = np.abs(h_vals)
        self.phase_characteristic = angle

    def calc_freq_characteristic_params(self, force: bool = False) -> None:
        if force is False and self.boundary_frequencies is not None and self.bandwidth is not None:
            return

        if self.amplitude_characteristic is None:
            self.calc_freq_characteristic(True)

        self.amplitude_characteristic_decibel = transform_data_to_db_scale(self.amplitude_characteristic)
        max_decibel = np.max(self.amplitude_characteristic_decibel)

        self.boundary_frequencies = np.ones(2, dtype=int) * -1
        self.boundary_decibel = max_decibel - 3

        for freq, decibel in enumerate(self.amplitude_characteristic_decibel):
            if self.boundary_frequencies[0] == -1 and decibel > self.boundary_decibel:
                self.boundary_frequencies[0] = freq
                continue

            if self.boundary_frequencies[0] != -1 and decibel < self.boundary_decibel:
                self.boundary_frequencies[1] = freq
                break

        self.bandwidth = self.boundary_frequencies[1] - self.boundary_frequencies[0]

    def get_decibels_for_boundary_frequencies(self) -> np.array:
        if self.amplitude_characteristic_decibel is None:
            self.calc_freq_characteristic_params(True)

        return np.array([
            self.amplitude_characteristic_decibel[self.boundary_frequencies[0]],
            self.amplitude_characteristic_decibel[self.boundary_frequencies[1]],
        ])

    def get_params(self) -> dict:
        if self.boundary_frequencies is None or self.bandwidth is None:
            self.calc_freq_characteristic_params(True)

        return {
            'max_freq': self.max_freq,
            'fs': self.fs,
            'boundary_frequencies': self.boundary_frequencies,
            'bandwidth': self.bandwidth,
        }
