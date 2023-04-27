from scipy.signal import iirfilter, freqs, sosfreqz
import numpy as np
from matplotlib import pyplot as plt

from Filter import Filter
from FilterVisualizer import FilterVisualizer

if __name__ == '__main__':
    fs = 48000
    fc = 1500
    coefficient_number = 1000

    sos = iirfilter(17, [50, 200], rs=60, analog=False,
                            btype='band', ftype='cheby2', fs=48000, output='sos')
    w, h = sosfreqz(sos, 2000, fs=48000)

    plt.figure()
    plt.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    plt.show()

    filter1 = Filter(20 * np.log10(np.maximum(abs(h), 1e-5)), 24000)
    # filter2 = Filter(iirfilter(coefficient_number, [300, 3500], fs=fs * 2, ftype='butter')[1], fs * 4)
    #
    print(filter1.get_params())
    FilterVisualizer.visualize_amplitude_db(filter1)
    FilterVisualizer.visualise_phase(filter1)

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
