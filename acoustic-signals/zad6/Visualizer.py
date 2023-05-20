from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt


class Visualizer:
    @staticmethod
    def visualize_spectrogram(signal: np.array) -> None:
        plt.figure(figsize=(20, 10))
        plt.specgram(signal, NFFT=4096, pad_to=3072)
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_spectrum(freqs: np.array, spectrum: np.array, is_in_decibel: bool = False) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(freqs, spectrum)
        plt.title('Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Level [dB]' if is_in_decibel else 'Amplitude')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_signal(signal: np.array, points: np.array = None, is_in_decibel: bool = False) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(signal)

        if points is not None:
            plt.scatter(points[:, 0], points[:, 1], c='orange')

        plt.title('Signal with formants' if points is not None else 'Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Level [dB]' if is_in_decibel else 'Amplitude')
        plt.show()
        plt.close()
