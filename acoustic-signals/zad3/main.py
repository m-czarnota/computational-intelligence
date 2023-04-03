import cmath
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import filtfilt, lfilter, firwin

from Filter import Filter
from Signal import Signal

FILTERS_DIR = './filters'


def transform_amplitude_to_db(signal: np.array) -> np.array:
    signal_max = np.max(signal)

    return 10 * np.log10(signal / signal_max)


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


def monophonize_signal_by_mean(signal: np.array) -> np.array:
    return signal.sum(axis=1) / 2


def apply_filter_to_signal(signal: Signal, filter: np.array) -> np.array:
    m_big = filter.shape[0]
    fc = boundary_freqs
    b = firwin(m_big, fc, fs=signal.samplerate)

    filtered_signal = np.zeros(signal.shape[0])

    for n in range(m_big - 1, signal.shape[0]):
        for m in range(m_big):
            filtered_signal[n] += b[m] * signal.data[n - m]

    return filtered_signal


if __name__ == '__main__':
    fir_data = np.loadtxt(f'{FILTERS_DIR}/Lab_03-Flt_01_CzM.txt')
    max_freq = 20000

    fir = Filter(fir_data, max_freq)
    fir.calc_freq_characteristic()

    # plt.figure()
    # plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic)
    # plt.title('Amplitude characteristics')
    # plt.xlabel('Hz')
    # plt.ylabel('H(jΩ)')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.arange(fir.max_freq), fir.phase_characteristic)
    # plt.title('Phase characteristics')
    # plt.xlabel('Hz')
    # plt.ylabel('H(jΩ)')
    # plt.show()

    fir.calc_freq_characteristic_params()

    plt.figure()
    plt.plot(np.full(fir.amplitude_characteristic_decibel.shape[0], fir.boundary_decibel))
    plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic_decibel)
    plt.scatter(fir.boundary_frequencies, fir.get_decibels_for_boundary_frequencies())
    plt.title(f'Amplitude characteristics in dB, bandwidth: {fir.bandwidth}Hz')
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.show()

    samplerate, data = wavfile.read('./HARD-WAGON.wav')
    data = data.astype(float)
    data = monophonize_signal_by_mean(data)
    signal_length = data.shape[0] / samplerate

    sound = Signal(data, samplerate)

    params = {
        # 'channels': data.shape[1],
        'frequency': f'{samplerate}Hz',
        'length': f'{signal_length}s',
    }
    print(params)

    filtered = apply_filter_to_signal(sound, fir.data)
    wavfile.write("hard-wagon-filtered.wav", samplerate, filtered.astype(np.int16))

    plt.figure()
    plt.plot(filtered)
    plt.show()




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

liczymy fft dla ramki o wielkości 4096
liczymy charakterystyke amplitudową, czyli moduł
i odkładamy do spektogramu
"""