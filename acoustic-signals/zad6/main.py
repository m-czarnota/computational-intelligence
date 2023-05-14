import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy import interpolate

from functions import monophonize

VOWELS_DIR = './vowels'


if __name__ == '__main__':
    fs, a = wavfile.read(f'{VOWELS_DIR}/a_C3_ep44.wav')
    a = monophonize(a)

    fx = np.abs(np.fft.fftfreq(a.shape[0], 1 / fs))[:a.shape[0] // 2 + 1]
    spectrum = np.abs(np.fft.rfft(a)) / (a.shape[0] // 2)
    max_oy = 1.05 * np.max(spectrum)
    y = 20 * np.log10(np.abs(spectrum))

    plt.figure()
    plt.specgram(a, NFFT=4096, pad_to=3072)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(fx, y)
    plt.show()
    plt.close()

    # spectrum = np.abs(np.fft.fft(a))
    # step = 10
    # mean_spectrum = [np.mean(spectrum[i:i + 10]) for i in range(0, spectrum.shape[0], step)]
    #
    # plt.figure()
    # plt.plot(mean_spectrum)
    # plt.show()
    # plt.close()

    max_range = 5000
    scale = 100
    y2 = y[:max_range + scale + 11]
    fx2 = fx[:max_range + scale + 11]

    wyj = []
    xx = []
    for i in range(len(y2) // scale):
        xx.append(fx2[i * scale])
        wyj.append(np.mean(y2[i * scale:(i + 1) * scale]))

    f = interpolate.interp1d(xx, wyj)
    w = f(fx2[:max_range])

    plt.figure()
    plt.plot(fx2[:max_range], w)
    plt.show()
    plt.close()



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