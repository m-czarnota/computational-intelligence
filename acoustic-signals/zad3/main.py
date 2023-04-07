from os.path import exists
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

from Filter import Filter
from Signal import Signal

FILTERS_DIR = './filters'


def read_filtered_sound() -> Optional[Signal]:
    filtered_sound_filename = './hard-wagon-filtered.wav'
    if exists(filtered_sound_filename) is False:
        return None

    fs, sound_data = wavfile.read(filtered_sound_filename)
    sound_data = sound_data.astype(float)

    return Signal(sound_data, fs)


if __name__ == '__main__':
    fir_data = np.loadtxt(f'{FILTERS_DIR}/Lab_03-Flt_01_CzM.txt')
    max_freq = 20000

    fir = Filter(fir_data, max_freq)
    print(fir.get_params())

    plt.figure()
    plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic)
    plt.title('Amplitude characteristics')
    plt.xlabel('Hz')
    plt.ylabel('H(jΩ)')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(np.arange(fir.max_freq), fir.phase_characteristic)
    plt.title('Phase characteristics')
    plt.xlabel('Hz')
    plt.ylabel('H(jΩ)')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(np.full(fir.amplitude_characteristic_decibel.shape[0], fir.boundary_decibel))
    plt.plot(np.arange(fir.max_freq), fir.amplitude_characteristic_decibel)
    plt.scatter(fir.boundary_frequencies, fir.get_decibels_for_boundary_frequencies())
    plt.title(f'Amplitude characteristics in dB, bandwidth: {fir.bandwidth}Hz')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Level [dB]')
    plt.show()
    plt.close()

    samplerate, data = wavfile.read('./HARD-WAGON.wav')
    sound = Signal(data.astype(float), samplerate)
    sound = sound.monophonize()
    print(sound.get_params())

    sound_filtered = read_filtered_sound()
    if sound_filtered is None:
        sound_filtered = sound.apply_filter(fir)
        wavfile.write("hard-wagon-filtered.wav", samplerate, sound_filtered.data.astype(np.int16))

    sound_fft = sound.transform_to_frequency_domain()
    sound_filtered_fft = sound_filtered.transform_to_frequency_domain()

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.abs(sound_fft.data))
    ax.set_title('Signal spectrum before apply filter')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.abs(sound_filtered_fft.data))
    ax.set_title('Signal spectrum after apply filter')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')

    plt.show()
    plt.close()

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.abs(sound_fft.data))
    ax.set_title('Signal spectrum before apply filter in logarithmic scale')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.set_yscale('log')

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.abs(sound_filtered_fft.data))
    ax.set_title('Signal spectrum after apply filter in logarithmic scale')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.set_yscale('log')

    plt.show()
    plt.close()

    sound_spectrogram = spectrogram(sound.data, sound.samplerate, nfft=4096)
    sound_filtered_spectrogram = spectrogram(sound_filtered.data, sound_filtered.samplerate, nfft=4096)

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    ax.specgram(sound.data, Fs=sound.samplerate, NFFT=4096)
    ax.set_title('Spectrogram for signal before apply filter')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    ax = plt.subplot(2, 1, 2)
    ax.specgram(sound_filtered.data, Fs=sound_filtered.samplerate, NFFT=4096)
    ax.set_title('Spectrogram for signal after apply filter')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    plt.show()
    plt.close()



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