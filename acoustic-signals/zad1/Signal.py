import numpy as np


class Signal:
    def __init__(self, signal: np.array, samplerate: int):
        self.signal: np.array = signal
        self.samplerate: int = samplerate

    def change_samplerate(self, new_samplerate: int):
        return self.__down_sample_signal(new_samplerate) if new_samplerate < self.samplerate else self.__up_sample_signal(new_samplerate)

    @property
    def shape(self):
        return self.signal.shape

    def __down_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.samplerate
        return self.signal[np.unique(np.around(np.arange(self.shape[0]) * ratio).astype(int)), :]

    def __up_sample_signal(self, samplerate: int) -> np.array:
        ratio = samplerate / self.samplerate
        new_samples = np.full((np.round(self.shape[0] * ratio).astype(int), self.shape[1]), np.nan)

        for dim in range(self.shape[1]):
            new_samples[np.around(np.arange(self.shape[0]) * ratio).astype(int), dim] = self.signal[:, dim]

            start_nan_index = -1

            for sample_iter, sample in enumerate(new_samples[:, dim]):
                if np.isnan(sample) and start_nan_index == -1:
                    start_nan_index = sample_iter
                    continue

                if not np.isnan(sample) and start_nan_index != -1:
                    mean_between_samples = np.mean([new_samples[start_nan_index - 1], new_samples[sample_iter]])
                    new_samples[start_nan_index - 1:sample_iter + 1] = mean_between_samples
                    start_nan_index = -1
                    continue

        return new_samples
