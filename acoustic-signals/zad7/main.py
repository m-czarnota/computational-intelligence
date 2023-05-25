from scipy.io import wavfile
from scipy.signal import convolve

from Signal import Signal
from Visualizer import Visualizer

if __name__ == '__main__':
    iir = Signal(*wavfile.read('./iir/01_CzM--Auto_Park.wav'))
    Visualizer.visualize_iir(iir.data)

    wiki = Signal(*wavfile.read('./signals/wiki_bioinformatics.wav'))
    wiki_convolved = Signal(wiki.fs, convolve(wiki.data, iir.data, mode='same'))
    wiki_convolved.save_to_file('./signals/wiki_convolved.wav')

    Visualizer.compare_normal_with_convolved(wiki, wiki_convolved, 'wiki')
    Visualizer.compare_normal_with_convolved_by_spectrogram(wiki, wiki_convolved, 'wiki')

    york = Signal(*wavfile.read('./signals/york_drums.wav'))
    york = york.change_samplerate(wiki.fs)
    york_convolved = Signal(wiki.fs, convolve(york.data, iir.data, mode='same'))
    york_convolved.save_to_file('./signals/york_convolved.wav')

    Visualizer.compare_normal_with_convolved(york, york_convolved, 'york')
    Visualizer.compare_normal_with_convolved_by_spectrogram(york, york_convolved, 'york')

    """
        zrobić resampling yorka
        sygnał pogłosu jest długi, bo ponad 3 sekundy i jeszcze dość mocny, także spektrogram jest mocno rozmazany
    """


