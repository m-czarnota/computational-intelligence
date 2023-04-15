from scipy.fft import fft
from scipy.signal import firwin
from typing_extensions import Self
import numpy as np

from Filter import Filter
from functions import transform_data_to_db_scale


class Signal:
    def __init__(self, data: np.array, samplerate: int):
        self.data: np.array = data
        self.samplerate: int = samplerate

        self.db_scale: np.array = None

    def change_samplerate(self, new_samplerate: int) -> Self:
        return Signal(
            self.__down_sample_signal(new_samplerate) if new_samplerate < self.samplerate else self.__up_sample_signal(new_samplerate),
            new_samplerate
        )

    def transform_to_db_scale(self, force: bool = False) -> np.array:
        if force is False and self.db_scale is not None:
            return self.db_scale

        self.db_scale = transform_data_to_db_scale(self.data)

        return self.db_scale

    def monophonize(self) -> Self:
        return Signal(
            self.data.sum(axis=1) / 2 if len(self.shape) > 1 and self.shape[1] > 1 else self.data,
            self.samplerate
        )

    def apply_filter(self, fir: Filter) -> Self:
        m_big = fir.shape[0]
        b = firwin(m_big, fir.boundary_frequencies, fs=self.samplerate)

        filtered_signal = np.zeros(self.shape[0])

        for n in range(m_big - 1, self.shape[0]):
            for m in range(m_big):
                filtered_signal[n] += b[m] * self.data[n - m]

        return Signal(filtered_signal, self.samplerate)

    def transform_to_frequency_domain(self) -> Self:
        return Signal(fft(self.data), self.samplerate)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def signal_length(self) -> float:
        return self.shape[0] / self.samplerate

    def __down_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.samplerate

        return self.data[np.unique(np.around(np.arange(self.shape[0]) * ratio).astype(int)), :]

    def __up_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.samplerate
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

    def get_params(self) -> dict:
        return {
            'frequency': f'{self.samplerate}Hz',
            'length': f'{self.signal_length}s',
        }
