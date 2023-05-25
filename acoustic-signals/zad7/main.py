import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import convolve


def monophonize(data: np.array) -> np.array:
    if len(data.shape) == 1:
        return data

    return data.sum(axis=1) / 2


if __name__ == '__main__':
    wiki_fs, wiki_data = wavfile.read('./01_Dry/wiki_bioinformatics.wav')
    wiki_data = monophonize(wiki_data)
    wiki_data = wiki_data.astype(np.float32)

    york_fs, york_data = wavfile.read('./01_Dry/york_drums.wav')
    york_data = monophonize(york_data)
    york_data = york_data.astype(np.float32)

    czm_fs, czm_data = wavfile.read('./02_IRs/01_Lexicon_480L/01_CzM--Auto_Park.wav')
    czm_data = monophonize(czm_data)
    czm_data = czm_data.astype(np.float32)

    plt.figure()
    plt.plot(czm_data)
    plt.title('Impulse response')
    plt.show()
    plt.close()

    wiki_convolved = convolve(wiki_data, czm_data, mode='same')
    york_convolved = convolve(york_data, czm_data, mode='same')

    plt.figure(1)
    plt.subplot(211)
    plt.title('Spectrogram of wiki signal')
    plt.specgram(wiki_data, Fs=wiki_fs)
    plt.subplot(212)
    plt.title('Spectrogram of convolved wiki signal')
    plt.specgram(wiki_convolved, Fs=wiki_fs)
    plt.show()
    plt.close()

    plt.figure(1)
    plt.subplot(211)
    plt.title('Spectrogram of york signal')
    plt.specgram(york_data, Fs=york_fs)
    plt.subplot(212)
    plt.title('Spectrogram of convolved york signal')
    plt.specgram(york_convolved, Fs=york_fs)
    plt.show()
    plt.close()

    """
        zrobić resampling yorka
        sygnał pogłosu jest długi, bo ponad 3 sekundy i jeszcze dość mocny, także spektrogram jest mocno rozmazany
    """


