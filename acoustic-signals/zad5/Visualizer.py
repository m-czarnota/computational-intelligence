import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft

from constants import octaves, terces


class Visualizer:
    @staticmethod
    def visualise_amplitude(data: np.array) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(np.abs(data))
        plt.title('Amplitude characteristics')
        plt.xlabel('Hz')
        plt.ylabel('H(jΩ)')
        plt.show()
        plt.close()

    @staticmethod
    def visualise_phase(data: np.array) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(np.unwrap(np.angle(data)))
        plt.title('Phase characteristics')
        plt.xlabel('Hz')
        plt.ylabel('H(jΩ)')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_spectrum(data: np.array) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(np.abs(fft(data)))
        plt.title('Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_spectrogram(data: np.array, fs: int = 2) -> None:
        plt.figure(figsize=(20, 10))
        plt.specgram(data, Fs=fs, NFFT=4096)
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_signal(data: np.array) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(data)
        plt.title('Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_octaves(data: np.array) -> None:
        spectrum = np.abs(fft(data))
        octaves_bands = np.empty(len(octaves))
        x = np.arange(1, len(octaves_bands) + 1)

        for octave_iter, octave in enumerate(octaves):
            octaves_bands[octave_iter] = np.mean(spectrum[int(octave[0]):int(octave[1])])

        plt.figure(figsize=(20, 10))
        plt.bar(x, octaves_bands, width=0.3)
        plt.title('Octave bands')
        plt.xticks(x)
        plt.xlabel('Octave number')
        plt.ylabel('Amplitude')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_terces(data: np.array) -> None:
        spectrum = np.abs(fft(data))
        terces_bands = np.empty(len(terces))
        x = np.arange(1, len(terces_bands) + 1)

        for terce_iter, terce in enumerate(terces):
            terces_bands[terce_iter] = np.mean(spectrum[int(terce[0]):int(terce[1])])

        plt.figure(figsize=(20, 10))
        plt.bar(x, terces_bands, width=0.3)
        plt.title('Terce bands')
        plt.xticks(x)
        plt.xlabel('Terce number')
        plt.ylabel('Amplitude')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison_amplitude(data1: np.array, data2: np.array):
        plt.figure(figsize=(20, 10))

        ax = plt.subplot(2, 1, 1)
        ax.plot(np.abs(data1))
        ax.set_title('Amplitude characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        ax = plt.subplot(2, 1, 2)
        ax.plot(np.abs(data2))
        ax.set_title('Amplitude characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison_phase(data1: np.array, data2: np.array) -> None:
        plt.figure(figsize=(20, 10))

        ax = plt.subplot(2, 1, 1)
        ax.plot(np.unwrap(np.angle(data1)))
        ax.set_title('Phase characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        ax = plt.subplot(2, 1, 2)
        ax.plot(np.unwrap(np.angle(data2)))
        ax.set_title('Phase characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison_spectrum(data1: np.array, data2_filtered: np.array, log_scale: bool = False) -> None:
        plt.figure(figsize=(20, 10))

        ax = plt.subplot(2, 1, 1)
        ax.plot(np.abs(fft(data1)))
        ax.set_title(f'Signal spectrum before apply filter {"in log scale" if log_scale else ""}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        if log_scale:
            ax.set_yscale('log')

        ax = plt.subplot(2, 1, 2)
        ax.plot(np.abs(fft(data2_filtered)))
        ax.set_title(f'Signal spectrum after apply filter {"in log scale" if log_scale else ""}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        if log_scale:
            ax.set_yscale('log')

        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison_spectrogram(data1: np.array, data2: np.array, fs: int) -> None:
        plt.figure(figsize=(20, 10))

        ax = plt.subplot(2, 1, 1)
        ax.specgram(data1, Fs=fs, NFFT=4096)
        ax.set_title('Spectrogram for signal before apply filter')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')

        ax = plt.subplot(2, 1, 2)
        ax.specgram(data2.data, Fs=fs, NFFT=4096)
        ax.set_title('Spectrogram for signal after apply filter')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')

        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison_signals(data1: np.array, data2: np.array) -> None:
        plt.figure(figsize=(20, 10))

        ax = plt.subplot(2, 1, 1)
        ax.plot(data1)
        ax.set_title('Signal before filtration')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')

        ax = plt.subplot(2, 1, 2)
        ax.plot(data2.data)
        ax.set_title('Signal after filtration')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')

        plt.show()
        plt.close()
