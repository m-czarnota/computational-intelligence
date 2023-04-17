import numpy as np
from matplotlib import pyplot as plt

from Signal import Signal


class SignalVisualiser:
    @staticmethod
    def visualise_comparation_spectrogram_signals(signal1: Signal, signal2: Signal, log_scale: bool = False):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(2, 1, 1)
        ax.plot(np.abs(signal1.data))
        ax.set_title('Signal spectrum before apply filter')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        if log_scale:
            ax.set_yscale('log')

        ax = plt.subplot(2, 1, 2)
        ax.plot(np.abs(signal2.data))
        ax.set_title('Signal spectrum after apply filter')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        if log_scale:
            ax.set_yscale('log')

        plt.show()
        plt.close()
