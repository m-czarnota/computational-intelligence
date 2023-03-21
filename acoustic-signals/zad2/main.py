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
    fs = 88000
    dt = 1 / fs
    t = np.arange(0, 0.0025 - dt, dt)
    F = 400
    y = np.sin(2 * np.pi * F * t)

    moves = [1e-5]

    sound_speed = 344
    distance = 0.16

    for move in moves:
        moved = np.sin(2 * np.pi * F * (t + move))

        plt.figure()
        plt.plot(t, y, label='original signal')
        plt.plot(t, moved, label='moved signal')
        plt.show()

        correlation = np.correlate(a=y, v=moved, mode='full')
        delta_t = t[np.argmax(correlation)] * dt
        a = np.arcsin(delta_t * sound_speed / distance)
        print(a)

    # for moved_signal_iter, moved_signal in enumerate(moved_signals):
    #     correlation = np.correlate(a=original_signal, v=moved_signal.signal, mode='full')
    #     # plt.figure()
    #     # plt.plot(correlation)
    #     # plt.title(f'Correlation between original signal and signal moved about {signal_moves[moved_signal_iter]}s')
    #     # plt.show()
    #
    #     delta_t = np.argmax(correlation) * signal_moves[moved_signal_iter]
    #     # a = np.power(np.sin(delta_t * sound_speed / distance), -1)
    #     a = np.arcsin(delta_t * sound_speed / distance)
    #     print(a)

"""
c - prędkość dźwięku w powietrzu 344 m/s
d - odległość między uszami w m
delta t - wzór jest w instrukcji

"""


