import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


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


def draw_signal_db(left_channel: np.array, right_channel: np.array) -> None:
    left_preview_db = 10 * np.log10(left_channel / max_freq_left)
    right_preview_db = 10 * np.log10(right_channel / max_freq_right)

    left_min_db = np.nanmin(left_preview_db[left_preview_db != -np.inf])
    right_min_db = np.nanmin(right_preview_db[right_preview_db != -np.inf])

    left_db_minus = np.ones(left_channel.shape) * left_min_db
    left_db_plus = np.ones(left_channel.shape) * left_min_db

    right_db_minus = np.ones(left_channel.shape) * right_min_db
    right_db_plus = np.ones(left_channel.shape) * right_min_db

    for amp_iter, (amp_left, amp_right) in enumerate(zip(left_channel, right_channel)):
        amp_left_db = 10 * np.log(amp_left / max_freq_left)
        amp_right_db = 10 * np.log(amp_right / max_freq_right)

        if amp_left >= 0:
            left_db_plus[amp_iter] = amp_left_db
        else:
            left_db_minus[amp_iter] = amp_left_db

        if amp_right >= 0:
            right_db_plus[amp_iter] = amp_right_db
        else:
            right_db_minus[amp_iter] = amp_right_db

    ax = plt.subplot(2, 1, 1)
    ax.plot(np.linspace(0, signal_length, left_channel.shape[0]), left_db_minus)
    ax.plot(np.linspace(0, signal_length, left_channel.shape[0]), left_db_plus)
    ax.set_title('Left channel')
    ax.set_xlabel('Time')
    ax.set_ylabel('dB')

    ax = plt.subplot(2, 1, 2)
    ax.plot(np.linspace(0, signal_length, right_channel.shape[0]), right_db_minus)
    ax.plot(np.linspace(0, signal_length, right_channel.shape[0]), right_db_plus)
    ax.set_title('Right channel')
    ax.set_xlabel('Time')
    ax.set_ylabel('dB')

    plt.show()


def draw_monofonic_signal(signal: np.array) -> None:
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


def monofonic_signal_by_mean(signal: np.array) -> np.array:
    return signal.sum(axis=1) / 2


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

    t = np.linspace(0., 1., samplerate)
    fs = 100

    amplitude = np.iinfo(np.int16).max
    modified_data = amplitude * np.sin(2. * np.pi * fs * t)
    wavfile.write("signal_saved_mono_16.wav", samplerate, modified_data.astype(np.int16))

    max_freq_left = np.max(data[:, 0])
    min_freq_left = np.min(data[:, 0])

    max_freq_right = np.max(data[:, 1])
    min_freq_right = np.min(data[:, 1])

    # draw_signal_amplitude(data[:, 0] / max_freq_left, data[:, 1] / max_freq_right)

    draw_signal_db(
        data[:, 0],
        data[:, 1]
    )
    # draw_signal_db(
    #     20 * np.log10(data[:, 0] / -0.3),
    #     20 * np.log10(data[:, 1] / -0.3)
    # )
    # draw_signal_db(
    #     20 * np.log10(data[:, 0] / -3),
    #     20 * np.log10(data[:, 1] / -3)
    # )

    monofonic_signal = monofonic_signal_by_mean(data)
    # draw_monofonic_signal(monofonic_signal)

    signal_with_removed_const_value = remove_const_value(data)
    # draw_signal_amplitude(signal_with_removed_const_value[:, 0], signal_with_removed_const_value[:, 1])


"""
normalizujemy do wartołści maksymalnej i jeszcze potem do zadanej wartości
składowa stała - wartość średnia sygnału, sygnał nie jest rozłożony równomiernie wzdłuż osi x

zad 3
trzeba zrobić interpolacje
robimy wspólą wielokrotność, czyli upsampling, czyli interpolacje znacznie większą
a potem zrobić decymacje, czyli usuwanie co drugiej próbki
wykres błędu średniokwadratowego
"""
