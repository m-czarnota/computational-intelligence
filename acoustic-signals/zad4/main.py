import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin

from Filter import Filter
from FilterVisualizer import FilterVisualizer
from Signal import Signal


def generate_filter_coefficients(coefficient_number: int, fs: int, fc: int) -> np.array:
    """
    :param coefficient_number:
    :param fs: sampling frequency
    :param fc: boundary frequency
    :return:
    """
    nyquist_rate = fs / 2
    cutoff_frequency = fc / nyquist_rate

    return firwin(coefficient_number, cutoff_frequency, window='hamming')

    coefficients = np.zeros(coefficient_number)

    for i in range(coefficient_number):
        nominator = np.sin(2 * np.pi * i * (fc / fs))
        denominator = (i * np.pi)
        coefficient = nominator / (denominator if denominator != 0 else 1)

        coefficients[i] = coefficient

    coefficients *= np.hamming(coefficient_number)

    return coefficients


if __name__ == '__main__':
    fs = 48000
    fc = 1500
    coefficient_number = 1000

    fir_coefficients = generate_filter_coefficients(coefficient_number, fs, fc)
    fir = Filter(fir_coefficients, int(fs / 2))
    print(fir.get_params())

    plt.figure()
    plt.plot(fir_coefficients)
    plt.show()
    plt.close()

    FilterVisualizer.visualise_amplitude(fir)
    FilterVisualizer.visualise_phase(fir)
