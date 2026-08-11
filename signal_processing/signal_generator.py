import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Signal parameters
# -----------------------------

SAMPLE_RATE = 1000
DURATION = 1.0
SIGNAL_FREQUENCY = 50

RANDOM_SEED = 42
SNR_LEVELS = [10, 5, 0, -5, -10]


# -----------------------------
# Generate clean signal
# -----------------------------

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


# -----------------------------
# Add controlled noise
# -----------------------------

def add_noise(signal, snr_db):

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=len(signal)
    )

    noisy_signal = signal + noise

    return noisy_signal  
# -----------------------------
# Generate noise-only sample
# -----------------------------

def generate_noise_only(signal_length, noise_power=1.0):

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=signal_length
    )

    return noise


# -----------------------------
# Generate signal + noise sample
# -----------------------------

def generate_signal_with_noise(snr_db):

    time, clean_signal = generate_clean_signal()

    noisy_signal = add_noise(
        clean_signal,
        snr_db
    )

    return time, noisy_signal


# -----------------------------
# Test the functions
# -----------------------------

if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    time, signal = generate_clean_signal()

    snr_db = 0

    noisy_signal = add_noise(
        signal,
        snr_db
    )

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        noisy_signal
    )

    plt.title(
        f"Noisy Synthetic Signal — SNR = {snr_db} dB"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()

    plt.show()
