import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from sklearn.metrics import mean_squared_error

from Signal import Signal


def draw_signal_amplitude(left: np.array, right: np.array) -> None:
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.linspace(0, signal_length, left.shape[0]), left)
    ax.set_title('Left channel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.linspace(0, signal_length, right.shape[0]), right)
    ax.set_title('Right channel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    plt.show()


def transform_amplitude_to_db(signal: np.array) -> np.array:
    left = signal[:, 0]
    right = signal[:, 1]

    max_left = np.max(left)
    max_right = np.max(right)

    return np.array([
        10 * np.log10(left / max_left),
        10 * np.log10(right / max_right)
    ]).T


def prepare_signal_to_draw_db(signal_db: np.array) -> np.array:
    left = signal_db[:, 0]
    right = signal_db[:, 1]

    left_min_db = np.nanmin(left[left != -np.inf])
    left_db = left.copy()
    left_db[np.where(np.isnan(left_db))] = left_min_db

    right_min_db = np.nanmin(right[right != -np.inf])
    right_db = right.copy()
    right_db[np.where(np.isnan(right_db))] = right_min_db

    return np.array([left_db, right_db]).T


def draw_signal_db(signal_db: np.array) -> None:
    prepared_signal = prepare_signal_to_draw_db(signal_db)
    left = prepared_signal[:, 0]
    right = prepared_signal[:, 1]

    left_min = np.min(left[left != -np.inf])
    right_min = np.min(right[right != -np.inf])

    plt.subplots_adjust(hspace=0)
    ax = plt.subplot(2, 1, 1)
    plt.title('Left channel dB')
    ax.plot(np.linspace(0, signal_length, left.shape[0]), left)
    # ax.set_yticks(np.arange(0, left_min, -20))
    ax.set_xticks([])
    ax.margins(0.05, 0)

    ax = plt.subplot(2, 1, 2)
    ax.invert_yaxis()
    # ax.set_position([0.125, 0.49, 0.775, 0.19])
    ax.plot(np.linspace(0, signal_length, left.shape[0]), left)
    ax.set_xlabel('Time')
    ax.set_ylabel('dB')
    # ax.set_yticks([np.arange(0, left_min, -20)])
    ax.margins(0.05, 0)

    plt.show()

    plt.subplots_adjust(hspace=0)
    ax = plt.subplot(2, 1, 1)
    plt.title('Right channel dB')
    ax.plot(np.linspace(0, signal_length, right.shape[0]), right, label='plus')
    # ax.set_yticks(np.arange(0, left_min_db, -20))
    ax.set_xticks([])
    ax.margins(0.05, 0)

    ax = plt.subplot(2, 1, 2)
    ax.invert_yaxis()
    # ax.set_position([0.125, 0.53, 0.775, 0.19])
    ax.plot(np.linspace(0, signal_length, right.shape[0]), right)
    ax.set_xlabel('Time')
    ax.set_ylabel('dB')
    # ax.set_yticks(np.arange(0, left_min_db, -20))
    ax.margins(0.05, 0)

    plt.show()


def draw_monophonic_signal(signal: np.array) -> None:
    plt.figure()
    plt.plot(np.linspace(0, signal_length, signal.shape[0]), signal)

    plt.title('Monofonic signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.show()


def remove_const_value(signal: np.array):
    mean_amplitude_left = np.mean(signal[:, 0])
    mean_amplitude_right = np.mean(signal[:, 1])

    return np.array([
        signal[signal[:, 0] != mean_amplitude_left, 0],
        signal[signal[:, 1] != mean_amplitude_right, 1],
    ]).T


def monophonize_signal_by_mean(signal: np.array) -> np.array:
    return signal.sum(axis=1) / 2


def normalize_signal(signal: np.array, to_value: float = None) -> np.array:
    left = signal[:, 0]
    right = signal[:, 1]

    max_left = np.max(left)
    max_right = np.max(right)

    return np.array([
        left / max_left,
        right / max_right
    ]).T


if __name__ == '__main__':
    samplerate, data = wavfile.read('signal.wav')
    data = data.astype(float)
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

    # saving signal to file
    t = np.linspace(0., 1., samplerate)
    fs = 100

    amplitude = np.iinfo(np.int16).max
    modified_data = amplitude * np.sin(2. * np.pi * fs * t)
    wavfile.write("signal_saved_mono_16.wav", samplerate, modified_data.astype(np.int16))

    # normalizing signal
    data = normalize_signal(data)
    # draw_signal_amplitude(data[:, 0], data[:, 1])

    # transforming normalized signal to db scale
    data_db = transform_amplitude_to_db(data)
    # draw_signal_db(data_db)

    # monophoning normalized signal
    monophonic_signal = monophonize_signal_by_mean(data)
    # draw_monophonic_signal(monophonic_signal)

    signal_with_removed_const_value = remove_const_value(data)
    # draw_signal_amplitude(signal_with_removed_const_value[:, 0], signal_with_removed_const_value[:, 1])

    signal = Signal(data, samplerate)
    changed = signal.change_samplerate(48000)
    returned = changed.change_samplerate(signal.samplerate)
    mse = mean_squared_error(signal.signal, returned.signal)
    print(f'MSE for upsampling = {mse}')

"""
normalizujemy do wartołści maksymalnej i jeszcze potem do zadanej wartości
składowa stała - wartość średnia sygnału, sygnał nie jest rozłożony równomiernie wzdłuż osi x

zad 3
trzeba zrobić interpolacje
robimy wspólą wielokrotność, czyli upsampling, czyli interpolacje znacznie większą
a potem zrobić decymacje, czyli usuwanie co drugiej próbki
wykres błędu średniokwadratowego
"""
