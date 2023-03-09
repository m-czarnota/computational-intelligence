import numpy as np
from matplotlib import pyplot as plt
from Signal import Signal

SIGNAL_LENGTH: float = 1


def generate_sinus(frequency: int, moving_interval_time: float = 0) -> np.array:
    samples_count = frequency * SIGNAL_LENGTH
    samples = np.linspace(0, SIGNAL_LENGTH, int(samples_count)) + moving_interval_time

    return np.sin(2 * np.pi * samples)


def move_signal(freq: int, moving_interval_time: float, visualize: bool = False):
    original = generate_sinus(freq)
    moved = generate_sinus(freq, moving_interval_time)

    if visualize:
        plt.figure()
        plt.plot(original, label='Original signal')
        plt.plot(moved, label=f'Moved signal with {moving_interval_time}s')
        plt.title('Original and moved sinus')
        plt.legend()
        plt.show()

    return Signal(moved, freq)


if __name__ == '__main__':
    signal_freq = 44100
    move_signal(signal_freq, 0, False)

    original_signal = generate_sinus(signal_freq)
    signal_moves = [0.0004, 0.0008, 0.0012, 0.0016, 0.002]
    moved_signals = [move_signal(signal_freq, interval_time, False) for interval_time in signal_moves]

    correlation = np.correlate(a=original_signal, v=moved_signals[-1].signal, mode='full')
    plt.figure()
    plt.plot(correlation)
    plt.show()

"""
c - prędkość dźwięku w powietrzu 340 m/s
d - odległość w m
delta t - wzór jest w instrukcji

"""


