import numpy as np
from matplotlib import pyplot as plt

from Filter import Filter


class FilterVisualizer:
    @staticmethod
    def visualise_amplitude(fir: Filter, params: dict = {}) -> None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(fir.max_freq), fir.get_amplitude_characteristic())
        ax.set_title('Amplitude characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        if 'text' in params.keys():
            ax.text(-0.15, 0.95, params['text'], transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))

        plt.show()
        plt.close(fig)

    @staticmethod
    def visualise_phase(fir: Filter, params: dict = {}) -> None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(fir.max_freq), fir.get_phase_characteristic())
        ax.set_title('Phase characteristics')
        ax.set_xlabel('Hz')
        ax.set_ylabel('H(jΩ)')

        if 'text' in params.keys():
            ax.text(-0.15, 0.95, params['text'], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))

        plt.show()
        plt.close(fig)

    @staticmethod
    def visualize_amplitude_db(fir: Filter) -> None:
        plt.figure(figsize=(20, 10))
        plt.plot(np.full(fir.get_amplitude_characteristic_decibel().shape[0], fir.boundary_decibel))
        plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic_decibel)
        plt.scatter(fir.boundary_frequencies, fir.get_decibels_for_boundary_frequencies())
        plt.title(f'Amplitude characteristics in dB, bandwidth: {fir.bandwidth}Hz')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Level [dB]')
        plt.show()
        plt.close()
