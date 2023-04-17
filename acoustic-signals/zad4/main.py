import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin

from Filter import Filter
from FilterVisualizer import FilterVisualizer
from Signal import Signal


def generate_filter_coefficients(coefficient_number: int, fs: int, fc: int, window: str = 'hamming') -> np.array:
    """
    :param coefficient_number:
    :param fs: sampling frequency
    :param fc: boundary frequency
    :return:
    """
    nyquist_rate = fs / 2
    cutoff_frequency = fc / nyquist_rate

    return firwin(coefficient_number, cutoff_frequency, window=window)

    coefficients = np.zeros(coefficient_number)

    for i in range(coefficient_number):
        nominator = np.sin(2 * np.pi * i * (fc / fs))
        denominator = (i * np.pi)
        coefficient = nominator / (denominator if denominator != 0 else 1)

        coefficients[i] = coefficient

    if window == 'hamming':
        window_function = np.hamming
    elif window == 'hanning':
        window_function = np.hanning
    else:
        window_function = np.blackman

    coefficients *= window_function(coefficient_number)

    return coefficients


def calc_impulse_response(fir: np.array):
    impulse_response = np.zeros(fir.shape)
    cutoff_frequency = fc / fs

    for i in range(fir.shape[0]):
        value = i + 1
        denominator = value - fir.shape[0] / 2

        if value != int(fir.shape[0] / 2):
            impulse_response[i] = np.sin(2 * np.pi * cutoff_frequency * denominator) / denominator
        else:
            impulse_response[i] = 2 * np.pi * cutoff_frequency

    plt.figure()
    plt.plot(impulse_response)
    plt.title('Filter impulse response')
    plt.xlabel('Coefficient number')
    plt.ylabel('Coefficient value')
    plt.show()
    plt.close()


if __name__ == '__main__':
    fs = 48000
    fc = 1500
    coefficient_number = 1000

    fir_coefficients = generate_filter_coefficients(coefficient_number, fs, fc)

    plt.figure()
    plt.plot(fir_coefficients)
    plt.show()
    plt.close()

    calc_impulse_response(fir_coefficients)

    fir = Filter(fir_coefficients, int(fs / 2))
    print(fir.get_params())



    FilterVisualizer.visualise_amplitude(fir)
    FilterVisualizer.visualise_phase(fir)
