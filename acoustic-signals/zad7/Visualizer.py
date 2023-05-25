import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from Signal import Signal


class Visualizer:
    @staticmethod
    def visualize_iir(data) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(data)
        plt.title('Impulse response')
        plt.xlabel('Coefficient number')
        plt.ylabel('Coefficient value')
        plt.show()
        plt.close()

    @staticmethod
    def compare_normal_with_convolved(normal: Signal, convolved: Signal, name: str) -> None:
        plt.figure(figsize=(20, 10))

        plt.subplot(211)
        plt.title(f'{name.capitalize()} signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, normal.length, normal.shape[0]), normal.data)

        plt.subplot(212)
        plt.title(f'Convolved {name.lower()} signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, convolved.length, convolved.shape[0]), convolved.data)

        plt.show()
        plt.close()

    @staticmethod
    def compare_normal_with_convolved_by_spectrogram(normal: Signal, convolved: Signal, name: str) -> None:
        plt.figure(figsize=(20, 10))

        plt.subplot(211)
        plt.title(f'Spectrogram of {name.lower()} signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        wiki_spectrum, _, _, _ = plt.specgram(normal.data, Fs=normal.fs)
        plt.colorbar()

        plt.subplot(212)
        plt.title(f'Spectrogram of convolved {name.lower()} signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        wiki_convolved_spectrum, _, _, _ = plt.specgram(convolved.data, Fs=convolved.fs)
        plt.colorbar()

        plt.show()
        plt.close()

        mse = mean_squared_error(wiki_spectrum, wiki_convolved_spectrum)
        print(f'mse for {name}: {mse}')
