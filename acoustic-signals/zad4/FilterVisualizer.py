import numpy as np
from matplotlib import pyplot as plt

from Filter import Filter


class FilterVisualizer:
    @staticmethod
    def visualise_amplitude(fir: Filter) -> None:
        plt.figure()
        plt.plot(np.arange(fir.max_freq), fir.get_amplitude_characteristic())
        plt.title('Amplitude characteristics')
        plt.xlabel('Hz')
        plt.ylabel('H(jΩ)')
        plt.show()
        plt.close()

    @staticmethod
    def visualise_phase(fir: Filter) -> None:
        plt.figure()
        plt.plot(np.arange(fir.max_freq), fir.get_phase_characteristic())
        plt.title('Phase characteristics')
        plt.xlabel('Hz')
        plt.ylabel('H(jΩ)')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_amplitude_db(fir: Filter) -> None:
        plt.figure()
        plt.plot(np.full(fir.get_amplitude_characteristic_decibel().shape[0], fir.boundary_decibel))
        plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic_decibel)
        plt.scatter(fir.boundary_frequencies, fir.get_decibels_for_boundary_frequencies())
        plt.title(f'Amplitude characteristics in dB, bandwidth: {fir.bandwidth}Hz')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Level [dB]')
        plt.show()
        plt.close()
