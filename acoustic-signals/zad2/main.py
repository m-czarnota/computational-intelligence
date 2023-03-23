import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from scipy.io import wavfile

SOUNDS_DIR = './sounds'


def generate_sinus(sample_rate: int = 96000, frequency: float = 4000, shift: float = 0.0) -> Tuple:
    dt = 1 / sample_rate
    t = np.arange(0, 0.0025 - dt, dt)

    return t, np.sin(2 * np.pi * frequency * (t + shift))


def experiment(shift_val: float, original_signal: np.array) -> None:
    time_moved, signal_moved = generate_sinus(sample_rate=sample_rate, shift=shift_val)  # generating moved signal
    wavfile.write(f"{SOUNDS_DIR}/signal_shifted_{shift_val}_mono_32.wav", sample_rate, time_moved.astype(np.int32))

    # visualisation
    plt.figure()
    plt.plot(time, original_signal, label='original signal')
    plt.plot(time, signal_moved, label=f'shifted signal about {shift_val}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    correlation = np.correlate(a=original_signal, v=signal_moved, mode='full')
    correlation_ratio = correlation.shape[0] / time_moved.shape[0]  # ratio to find where should be real value

    delta_t = time_moved[int(np.argmax(correlation) / correlation_ratio)] * (1 / sample_rate)
    a = np.arcsin(delta_t * sound_speed / distance)

    print(f'Angle for moved signal: {a}')


if __name__ == '__main__':
    sample_rate = 96000
    moves = [0.00002, 0.00004, 0.00008, 0.00012, 0.00016, 0.0002]  # moves signal about part of second amount

    time, signal = generate_sinus(sample_rate=sample_rate)
    wavfile.write(f"{SOUNDS_DIR}/signal_original_mono_32.wav", sample_rate, signal.astype(np.int32))

    sound_speed = 344  # sound speed in air with temperature 20°C, m/s
    distance = 0.21  # radius of head, distance between ears in meters

    for move in moves:
        experiment(move, signal)

"""
c - prędkość dźwięku w powietrzu 344 m/s
d - odległość między uszami w m
delta t - wzór jest w instrukcji

"""


