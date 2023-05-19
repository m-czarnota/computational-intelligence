from typing import Tuple
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy import interpolate

from Visualizer import Visualizer
from functions import monophonize

VOWELS_DIR = './vowels'


def interpolate_signal(signal: np.array, max_range: int, step: int) -> Tuple:
    signal_ranged = signal[:max_range + step + 11]
    freqs_ranged = np.arange(signal.shape[0])[:max_range + step + 11]

    output = []
    input = []

    for i in range(len(signal_ranged) // step):
        input.append(freqs_ranged[i * step])
        output.append(np.mean(signal_ranged[i * step:(i + 1) * step]))

    interpolated = interpolate.interp1d(input, output)

    return freqs_ranged, interpolated(freqs_ranged[:max_range])


def find_peaks(signal: np.array, first: bool = False) -> dict:
    peaks = {}

    for index in range(1, signal.shape[0] - 1):
        sample = signal[index]
        if sample < -50:
            continue

        prev_sample = signal[index - 1]
        next_sample = signal[index + 1]

        if sample > prev_sample and sample > next_sample:
            peaks[index] = sample

            if first:
                break

    if len(peaks) == 0:
        index = signal.shape[0] - 1
        peaks[index] = signal[index]

    return peaks


def calc_formants(signal: np.array) -> dict:
    peaks = find_peaks(signal)
    print(peaks)
    keys = list(peaks.keys())

    formants = {}
    current_iter = 0

    while current_iter < len(peaks):
        current_freq = keys[current_iter]
        freqs = [current_freq]

        while current_iter < len(peaks) - 1:
            current_iter += 1
            freq = keys[current_iter]

            # groups will not be further away than 1000 Hz
            if freq > current_freq + 1000:
                current_iter -= 1
                break

            freqs.append(keys[current_iter])

        formant_index = int(np.mean(freqs))
        formants[formant_index] = signal[formant_index]
        current_iter += 1

    return formants


def calc_auto_correlation(signal: np.array) -> np.array:
    result = np.correlate(signal, signal, mode='full')

    return result[len(result) // 2:]


def calc_f0(signal: np.array, fs: int) -> int:
    auto_correlation = calc_auto_correlation(signal)
    k0 = list(find_peaks(auto_correlation, True).keys())[0]

    return fs / k0


def find_bandwidth(signal: np.array, formant_freq: int) -> np.array:
    bandwidth = np.ones(2) * -1
    formant_amplitude = signal[formant_freq]
    searched_amplitude = formant_amplitude - 3
    changing_range = 500

    start_point = formant_freq - changing_range if formant_freq >= changing_range else 0
    end_point = formant_freq + changing_range if formant_freq <= signal.shape[0] - changing_range else signal.shape[0]

    for i in range(start_point, end_point):
        sample = signal[i]

        if bandwidth[0] == -1 and sample >= searched_amplitude:
            bandwidth[0] = i
            continue

        if bandwidth[0] != -1 and sample <= searched_amplitude:
            bandwidth[1] = i
            break

    return bandwidth[1] - bandwidth[0]


def calc_frequency_response_for_formants(formants: dict, fs: int, alpha: float = 1.1) -> np.array:
    return np.array([1 / (1 - 2 * alpha * np.cos(2 * np.pi * freq / fs) + alpha ** 2) for freq in formants.keys()])


def calc_filter(signal: np.array, formants: dict, f0: int) -> np.array:
    formants_bandwidth = np.array([find_bandwidth(signal, formant) for formant in formants])

    t = np.power(f0, -1)
    output = np.empty_like(signal)
    output[:3] = signal[:3]

    for sample_iter in range(2, signal.shape[0]):
        elements = np.empty(len(formants))

        for formant_iter, (formant_freq, formant_amplitude) in enumerate(formants.items()):
            formant_bandwidth = formants_bandwidth[formant_iter]

            c = -np.exp(-2 * np.pi * formant_bandwidth * t)
            b = 2 * np.exp(-np.pi * formant_bandwidth * t) * np.cos(2 * np.pi * formant_freq * t)
            a = 1 - b - c
            h = (a if not np.isinf(a) and not np.isnan(a) else 0) * signal[sample_iter] + \
                (b if not np.isinf(b) and not np.isnan(b) else 0) * output[sample_iter - 1] + \
                (c if not np.isinf(c) and not np.isnan(c) else 0) * output[sample_iter - 2]

            elements[formant_iter] = h

        output[sample_iter] = np.nansum(elements)

    return output


if __name__ == '__main__':
    # fs, a = wavfile.read(f'{VOWELS_DIR}/a_C3_ep44.wav')
    fs, a = wavfile.read(f'mama.wav')
    a = monophonize(a)

    spectrum_freqs = np.abs(np.fft.fftfreq(a.shape[0], 1 / fs))
    spectrum = np.abs(np.fft.fft(a))
    spectrum_decibel = 20 * np.log10(spectrum / np.max(spectrum))

    Visualizer.visualize_spectrogram(a)
    Visualizer.visualize_spectrum(spectrum_freqs, spectrum_decibel, True)

    ranged_freqs, ranged_spectrum = interpolate_signal(spectrum_decibel, 10000, 100)
    Visualizer.visualize_signal(ranged_spectrum, is_in_decibel=True)

    formants = calc_formants(ranged_spectrum)
    print('Formants [Hz: dB]:\n', formants)

    f0 = calc_f0(ranged_spectrum, fs)
    print(f'Frequency f0: {f0}Hz')

    Visualizer.visualize_signal(ranged_spectrum, np.array([list(formants.keys()), list(formants.values())]).T, True)

    formants = {
        650: -27.575885497191074,
        3000: ranged_spectrum[3000],
        6500: -40.25234629438399,
    }
    print('Formants [Hz: dB]:\n', formants)
    Visualizer.visualize_signal(ranged_spectrum, np.array([list(formants.keys()), list(formants.values())]).T, True)

    frequency_reponse_formants = calc_frequency_response_for_formants(formants, fs)
    print('Frequency responses for formants:\n', frequency_reponse_formants)

    filtr = calc_filter(ranged_spectrum, formants, f0)
    print(filtr)

    plt.figure(figsize=(20, 10))
    plt.plot(filtr)
    plt.show()



"""
nagrać słowo i wyciąć samogłoski ręcznie
poobserwować widomo, spektrogram, nie powinno się wiele różnić

podział na podpasma, uśrednienie, odczytanie prosto z wykresu

formanty - pewne obszary częsottliwości, w której będziemy obserowwali podbicie charakterystyki
czyli uzyskany charakterystyke, która będzie nierównomiernie w danym paśmie częstotliwości
będzie kilka maksimów lokalnych (formantów), więc trzeba to będzie uśrednić widmo

czwarty pkt nie robić
zostają 3 wzory
"""