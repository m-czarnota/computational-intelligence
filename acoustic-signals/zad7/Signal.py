from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from functions import monophonize


class Signal:
    def __init__(self, fs: int, data: np.array):
        self.data: np.array = monophonize(data).astype(np.float32)
        self.fs: int = fs

    def change_samplerate(self, new_fs: int) -> Signal:
        return Signal(
            new_fs,
            self.__down_sample_signal(new_fs) if new_fs < self.fs else self.__up_sample_signal(new_fs),
        )

    def save_to_file(self, filepath: str) -> None:
        wavfile.write(filepath, self.fs, self.data.astype(np.int32))

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def length(self) -> float:
        return self.shape[0] / self.fs

    def __down_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.fs

        return self.data[np.unique(np.around(np.arange(self.shape[0]) * ratio).astype(int))]

    def __up_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.fs
        new_samples = np.full((np.round(self.shape[0] * ratio).astype(int), self.shape[1]), np.nan)

        for dim in range(self.shape[1]):
            new_samples[np.around(np.arange(self.shape[0]) * ratio).astype(int), dim] = self.data[:, dim]

            start_nan_index = -1

            for sample_iter, sample in enumerate(new_samples[:, dim]):
                if np.isnan(sample) and start_nan_index == -1:
                    start_nan_index = sample_iter
                    continue

                if not np.isnan(sample) and start_nan_index != -1:
                    mean_between_samples = np.mean([new_samples[start_nan_index - 1, dim], new_samples[sample_iter, dim]])
                    new_samples[start_nan_index - 1:sample_iter + 1] = mean_between_samples
                    start_nan_index = -1
                    continue

        return new_samples
