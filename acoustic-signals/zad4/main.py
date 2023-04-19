import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin

from Filter import Filter
from FilterVisualizer import FilterVisualizer


def generate_filter_coefficients(coefficient_number: int, fs: int, fc: int, window: str = 'hamming') -> np.array:
    """
    :param window:
    :param coefficient_number:
    :param fs: sampling frequency
    :param fc: boundary frequency
    :return:
    """
    nyquist_rate = fs / 2
    cutoff_frequency = fc / nyquist_rate

    return firwin(coefficient_number, cutoff_frequency, window=window, pass_zero='lowpass')


def calc_impulse_response(fir: np.array, filter_params: dict):
    impulse_response = np.zeros(fir.shape)
    cutoff_frequency = filter_params["fc"] / filter_params["fs"]

    for i in range(fir.shape[0]):
        value = i + 1
        denominator = value - fir.shape[0] / 2

        if value != int(fir.shape[0] / 2):
            impulse_response[i] = np.sin(2 * np.pi * cutoff_frequency * denominator) / denominator
        else:
            impulse_response[i] = 2 * np.pi * cutoff_frequency

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(impulse_response)
    ax.set_title('Filter impulse response')
    ax.set_xlabel('Coefficient number')
    ax.set_ylabel('Coefficient value')
    ax.text(-0.15, 0.95, filter_params['info'], transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
    plt.show()
    plt.close(fig)


def plots_visual_comparison() -> None:
    fs = 48000
    fc = 3800
    coefficient_number = 1000

    for window in ['blackman', 'hann', 'hamming']:
        fir_coefficients = generate_filter_coefficients(coefficient_number, fs, fc, window=window)
        info = '\n'.join((
            f'window: {window}',
            f'coefficients: {coefficient_number}',
            f'fs: {fs}',
            f'fc: {fc}',
        ))

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(fir_coefficients)
        ax.set_title('Generated filter')
        ax.set_xlabel('Coefficient number')
        ax.set_ylabel('Coefficient value')
        ax.text(-0.15, 0.95, info, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        plt.show()
        plt.close(fig)

        calc_impulse_response(fir_coefficients, {'fs': fs, 'fc': fc, 'info': info})

        fir = Filter(fir_coefficients, int(fs / 2))
        print(fir.get_params())

        FilterVisualizer.visualise_amplitude(fir, {'text': info})
        FilterVisualizer.visualise_phase(fir, {'text': info})


def compare_characteristics() -> None:
    fs = 48000
    fc = 3800
    window = 'hamming'

    for coefficient_number in [100, 500, 1000, 2000, 5000]:
        fir_coefficients = generate_filter_coefficients(coefficient_number, fs, fc, window=window)
        info = '\n'.join((
            f'window: {window}',
            f'coefficients: {coefficient_number}',
            f'fs: {fs}',
            f'fc: {fc}',
        ))

        fir = Filter(fir_coefficients, int(fs / 2))
        print(fir.get_params())

        FilterVisualizer.visualise_amplitude(fir, {'text': info})
        FilterVisualizer.visualise_phase(fir, {'text': info})


if __name__ == '__main__':
    plots_visual_comparison()
    compare_characteristics()
