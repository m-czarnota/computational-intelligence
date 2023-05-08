from typing import Iterable, Tuple

from scipy.io import wavfile
from scipy.signal import iirfilter, freqs, sosfreqz, sosfiltfilt, butter
import numpy as np
from matplotlib import pyplot as plt

from Filter import Filter
from FilterTypeEnum import FilterTypeEnum
from IirTypeEnum import IirTypeFilter
from FilterVisualizer import FilterVisualizer
from functions import monophonize


def generate_filter(fs: int = 48000, fc: Iterable = [50, 200], order: int = 17,
                    btype: FilterTypeEnum = FilterTypeEnum.BANDPASS, ftype: IirTypeFilter = IirTypeFilter.BUTTERWORTH) -> Tuple:
    sos = iirfilter(order, fc, rs=60, analog=False,
                    btype=btype.value, ftype=ftype.value, fs=fs, output='sos')
    w, h = sosfreqz(sos, fs, fs=fs)

    return h, sos


if __name__ == '__main__':
    # plt.figure()
    # plt.plot(generate_filter(48000, [10000, 15000], 17, btype=FilterTypeEnum.BANDPASS, ftype=IirTypeFilter.CHEBYSHEV_2))
    # plt.show()

    filter1_data, sos_filter1 = generate_filter(24000, [2500], 29, btype=FilterTypeEnum.LOWPASS, ftype=IirTypeFilter.CHEBYSHEV_2)
    # plt.figure()
    # plt.plot(np.abs(filter1_data))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.unwrap(np.angle(filter1_data)))
    # plt.show()

    rng = np.random.default_rng()
    n = 201
    t = np.linspace(0, 1, n)
    x = 1 + (t < 0.5) - 0.25 * t ** 2 + 0.05 * rng.standard_normal(n)

    fs, signal_data = wavfile.read('./signal.wav')
    signal_data_monophonized = monophonize(signal_data)
    y = sosfiltfilt(sos_filter1, signal_data_monophonized)


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
