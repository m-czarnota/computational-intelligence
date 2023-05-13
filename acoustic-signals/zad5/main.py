from typing import Iterable, Tuple
from scipy.io import wavfile
from scipy.signal import iirfilter, freqs, sosfreqz, sosfiltfilt, firwin
import numpy as np

from Filter import Filter
from FilterTypeEnum import FilterTypeEnum
from FilterVisualizer import FilterVisualizer
from IirTypeEnum import IirTypeFilter
from Visualizer import Visualizer
from functions import monophonize


def generate_filter(fs: int = 48000, fc: Iterable = [50, 200], order: int = 17,
                    btype: FilterTypeEnum = FilterTypeEnum.BANDPASS, ftype: IirTypeFilter = IirTypeFilter.BUTTERWORTH) -> Tuple:
    sos = iirfilter(order, fc, rs=60, analog=False,
                    btype=btype.value, ftype=ftype.value, fs=fs, output='sos')
    w, h = sosfreqz(sos, fs, fs=fs)

    return h, sos


def compare_signal_with_filtered(signal_original: np.array, signal_filtered: np.array, fs: int) -> None:
    Visualizer.visualize_comparison_signals(signal_original, signal_filtered)
    Visualizer.visualize_comparison_spectrum(signal_original, signal_filtered)
    Visualizer.visualize_comparison_spectrum(signal_original, signal_filtered, log_scale=True)
    Visualizer.visualize_comparison_spectrogram(signal_original, signal_filtered, fs)


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


if __name__ == '__main__':
    filter_data1, sos_filter1 = generate_filter(48000, [10000, 15000], 17, btype=FilterTypeEnum.BANDPASS, ftype=IirTypeFilter.CHEBYSHEV_2)
    # Visualizer.visualise_amplitude(filter_data1)
    # Visualizer.visualise_phase(filter_data1)

    filter_data2, sos_filter2 = generate_filter(24000, [2500], 29, btype=FilterTypeEnum.LOWPASS, ftype=IirTypeFilter.CHEBYSHEV_2)
    # Visualizer.visualise_amplitude(filter_data2)
    # Visualizer.visualise_phase(filter_data2)

    # fir_fs = 48000
    # fir_coefficients = generate_filter_coefficients(1000, fir_fs, 3800, window='blackman')
    # fir = Filter(fir_coefficients, int(fir_fs / 2))
    # FilterVisualizer.visualise_amplitude(fir)
    # FilterVisualizer.visualise_phase(fir)

    fs, signal_data = wavfile.read('./signal.wav')
    signal_data_monophonized = monophonize(signal_data)
    signal_filtered = sosfiltfilt(sos_filter2, signal_data_monophonized)

    # compare_signal_with_filtered(signal_data_monophonized, signal_filtered, fs=fs)

    Visualizer.visualize_octaves(signal_filtered)
    Visualizer.visualize_terces(signal_filtered)


"""
dwa filtry, z czego jeden podobny do tego z poprzednich zajęć

charakterystyka butterwalta jest charakterystyką opadającą w dół

charakterystyka czybyszewa oparta na wielomianach czybyszewa
ma ona zafalowania w paśmie przepustowym, czyli nie jest płaska
dzięki temu stromość charakterystyki uzyskujemy większą
możemy mieć charakterystykę czybyszewa drugiego stopnia

zależy nam aby charakterystyka w paśmie przepustowym była jak najbardziej płaska
porównanie: robimy to samo dla obu filtrów noi, a potem porównujemy soi i noi

pasma oktawowe - wykres słupkowy
każdy słupek będzie oznaczał sumę wartości poszczegónych prążków
oktawa - jednakowa odległość pomiędzy punktami na osi x odpowieda dwukrotności 
oktawa - stosunek częstotliwości 1:2, czyli: 1000, 2000, 4000
tercjowe - dzielimy oktawe na 3 części
można unormować, ale nie trzeba
wziąć tabelkę z wikipedii
"""
