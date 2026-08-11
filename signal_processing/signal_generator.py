import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 1000
DURATION = 1.0
SIGNAL_FREQUENCY = 50


def generate_clean_signal():
    time = np.arange(
        0,
        DURATION,
        1 / SAMPLE_RATE
    )

    signal = np.sin(
        2 * np.pi * SIGNAL_FREQUENCY * time
    )

    return time, signal


if __name__ == "__main__":

    time, signal = generate_clean_signal()

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)

    plt.title("Clean Synthetic Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
