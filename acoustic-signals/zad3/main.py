import cmath
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

FILTERS_DIR = './filters'


def transform_amplitude_to_db(signal: np.array) -> np.array:
    signal_max = np.max(signal)

    return 10 * np.log10(signal_max / signal_max)


def calc_characteristic(signal_filter: np.array, amplitude: bool = True, phase: bool = True) -> list:
    h_vals = []
    angle = []

    for freq in range(max_freq):
        omega = 2 * np.pi * freq / fs
        h = 0
        # h = np.sum(signal_filter) * np.exp(-1j * omega * np.arange(signal_filter.shape[0]))

        for fir_iter, fir_val in enumerate(signal_filter):
            h += fir_val * np.exp(-1j * omega * fir_iter)

        h_vals.append(h)
        angle.append(np.degrees(cmath.phase(h)))

    module = np.abs(h_vals)
    results = []

    if amplitude:
        results.append(module)
    if phase:
        results.append(angle)

    return results


if __name__ == '__main__':
    fir = np.loadtxt(f'{FILTERS_DIR}/Lab_03-Flt_01_CzM.txt')
    max_freq = 20000
    fs = max_freq * 2

    amplitude, phase = calc_characteristic(fir)

    plt.figure()
    plt.plot(np.arange(max_freq), amplitude)
    plt.title('Amplitude characteristics')
    plt.xlabel('Hz')
    plt.ylabel('H(jΩ)')
    plt.show()

    plt.figure()
    plt.plot(np.arange(max_freq), phase)
    plt.title('Phase characteristics')
    plt.xlabel('Hz')
    plt.ylabel('H(jΩ)')
    plt.show()

    decibel_module = transform_amplitude_to_db(module)



"""
parametry:
    częstotliwość graniczna - w skali decybelowej od maxa odjąć -3dB i mamy dwie częstotliwości
    charakterystyka częstotliwościowa amplitudowa:
        liczbe zespoloną można prezdstawić jako moduł i kąt 
        moduł to jest długość promienia
        charakterystyka fazowa to jest kąt w funkcji częstotliwości
        charakterystyka amplitudowa to jest moduł
    
    filtracja w dziedzinie czasu: b i m są naszymi współczynnikami
    zastosować splot
    x(n - m) to jest nasz dźwięk. to musi być przesuwanie tak jak w splocie
    
    trzeba wyciągnąć moduł, to jest charakterystyka częstotliwościowa
    a potem trzeba wyciągnąć kąt i to jest charakterystyka fazowa
    
    przejść po wszystkich współczynnikach
    omega to jest 2*pi*f
    to f jest argumentem naszej funkcji zespolonej
    a(omega) + jb(omega)
    dla każdego omega zakładamy f od 0 do 20KhZ, liczymy dla każdego kroku i wylicza się kąt. a f zapisuje się gdzieś do listy
    to trzeba dwie pętle z krokiem 1 albo większym, tak żeby nie liczyło się zbyt długo
    charakterystyki częstotliwościowe przedstawiane są skali logarytmicznej na osi X i osi Y

pętla robić od 0 do 20 kHz i 
wykres dla osi X mamy dla omega od 0 do 1
przyjmijmy, że nasz częstotliwość wynosi 48kHz
zgodnie z Nyquistem nasze pasmo ma zakres (0; 24)kHz
"""