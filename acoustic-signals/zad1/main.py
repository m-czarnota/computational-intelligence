import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


def draw_signals(scale: str = 'line') -> None:
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.linspace(0, signal_length, data.shape[0]), data[:, 0] / max_freq_left)
    ax.set_title('Left channel')
    if scale != 'line':
        ax.set_yscale('log')

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.linspace(0, signal_length, data.shape[0]), data[:, 1] / max_freq_right)
    ax.set_title('Right channel')
    if scale != 'line':
        ax.set_yscale('log')

    plt.show()


def normalize(value_db) -> None:
    pass


if __name__ == '__main__':
    samplerate, data = wavfile.read('signal.wav')
    signal_length = data.shape[0] / samplerate

    params = {
        'channels': data.shape[1],
        'frequency': f'{samplerate}Hz',
        'length': f'{signal_length}s',
    }
    print(params)

    # time = np.linspace(0., signal_length, data.shape[0])
    # plt.plot(time, data[:, 0], label="Left channel")
    # plt.plot(time, data[:, 1], label="Right channel")
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.show()

    t = np.linspace(0., 1., samplerate)
    fs = 100

    amplitude = np.iinfo(np.int16).max
    modified_data = amplitude * np.sin(2. * np.pi * fs * t)
    wavfile.write("signal_saved_mono_16.wav", samplerate, modified_data.astype(np.int16))

    max_freq_left = np.max(data[:, 0])
    max_freq_right = np.max(data[:, 1])

    draw_signals()
    draw_signals('log')

    power_db_left = 20 * np.log10(data[:, 0] / max_freq_left)
    power_db_right = 20 * np.log10(data[:, 1] / max_freq_right)

"""
składowa stała - wartość średnia sygnału, sygnał nie jest rozłożony równomiernie wzdłuż osi x

zad 3
trzeba zrobić interpolacje
robimy wspólą wielokrotność, czyli upsampling, czyli interpolacje znacznie większą
a potem zrobić decymacje, czyli usuwanie co drugiej próbki
wykres błędu średniokwadratowego
"""
