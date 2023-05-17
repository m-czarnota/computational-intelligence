import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy import interpolate

from functions import monophonize

VOWELS_DIR = './vowels'


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


def calc_filter(signal: np.array, formants: dict, f0: int):
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
            h = a * signal[sample_iter] + b * output[sample_iter - 1] + output[sample_iter - 2]

            elements[formant_iter] = h

        # alpha = 1.1
        # output[sample_iter] = np.sum(elements) / (1 - 2 * alpha * np.cos(2 * np.pi * sample_iter / fs) + alpha ** 2)
        output[sample_iter] = np.sum(elements)

    return output


if __name__ == '__main__':
    fs, a = wavfile.read(f'{VOWELS_DIR}/a_C3_ep44.wav')
    a = monophonize(a)

    spectrum_freqs = np.abs(np.fft.fftfreq(a.shape[0], 1 / fs))
    spectrum = np.abs(np.fft.fft(a))
    spectrum_decibel = 20 * np.log10(spectrum / np.max(spectrum))

    plt.figure()
    plt.specgram(a, NFFT=4096, pad_to=3072)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(spectrum_freqs, spectrum_decibel)
    plt.show()
    plt.close()

    max_range = 5000
    scale = 100
    y2 = spectrum_decibel[:max_range + scale + 11]
    fx2 = np.arange(spectrum_decibel.shape[0])[:max_range + scale + 11]

    wyj = []
    xx = []
    for i in range(len(y2) // scale):
        xx.append(fx2[i * scale])
        wyj.append(np.mean(y2[i * scale:(i + 1) * scale]))

    f = interpolate.interp1d(xx, wyj)
    w = f(fx2[:max_range])

    peaks = find_peaks(w)
    print(peaks)
    formants = calc_formants(w)
    print(formants)
    f0 = calc_f0(w, fs)
    print(f0)

    plt.figure()
    # plt.plot(fx2[:max_range], w)
    plt.plot(w)
    plt.scatter(formants.keys(), formants.values(), c='orange')
    plt.show()
    plt.close()

    filtr = calc_filter(w, formants, f0)
    print(filtr)
    plt.figure()
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