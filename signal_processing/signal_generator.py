import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

SAMPLE_RATE = 1000
DURATION = 1.0
SIGNAL_FREQUENCY = 50

RANDOM_SEED = 42

# SNR levels used in the experiment
SNR_LEVELS = [10, 5, 0, -5, -10]


# ============================================================
# 1. GENERATE CLEAN SIGNAL
# ============================================================

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


# ============================================================
# 2. ADD NOISE TO SIGNAL
# ============================================================

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


# ============================================================
# 3. CALCULATE NOISE POWER
# ============================================================

def calculate_noise_power(signal, snr_db):

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    return noise_power


# ============================================================
# 4. GENERATE NOISE-ONLY SAMPLE
# Class 0
# ============================================================

def generate_noise_only(signal_length, noise_power):

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=signal_length
    )

    return noise


# ============================================================
# 5. GENERATE SIGNAL + NOISE SAMPLE
# Class 1
# ============================================================

def generate_signal_with_noise(snr_db):

    time, clean_signal = generate_clean_signal()

    noisy_signal = add_noise(
        clean_signal,
        snr_db
    )

    return time, noisy_signal


# ============================================================
# 6. GENERATE COMPLETE DATASET
# ============================================================

def generate_dataset(samples_per_class_per_snr=10):

    dataset = []

    sample_id = 0

    for snr_db in SNR_LEVELS:

        # ----------------------------------------------------
        # Class 0: Noise only
        # ----------------------------------------------------

        for _ in range(samples_per_class_per_snr):

            _, clean_signal = generate_clean_signal()

            noise_power = calculate_noise_power(
                clean_signal,
                snr_db
            )

            noise = generate_noise_only(
                signal_length=len(clean_signal),
                noise_power=noise_power
            )

            dataset.append({
                "sample_id": sample_id,
                "snr_db": snr_db,
                "signal": noise,
                "label": 0
            })

            sample_id += 1

        # ----------------------------------------------------
        # Class 1: Signal + Noise
        # ----------------------------------------------------

        for _ in range(samples_per_class_per_snr):

            time, noisy_signal = generate_signal_with_noise(
                snr_db
            )

            dataset.append({
                "sample_id": sample_id,
                "snr_db": snr_db,
                "signal": noisy_signal,
                "label": 1
            })

            sample_id += 1

    return dataset


# ============================================================
# 7. TEST CODE
# ============================================================

if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    # Generate a small test dataset
    test_dataset = generate_dataset(
        samples_per_class_per_snr=2
    )

    print("===================================")
    print("DATASET TEST")
    print("===================================")

    print(
        "Total samples:",
        len(test_dataset)
    )

    print(
        "Signal length:",
        len(test_dataset[0]["signal"])
    )

    print("\nFirst sample:")

    print(
        "Sample ID:",
        test_dataset[0]["sample_id"]
    )

    print(
        "SNR:",
        test_dataset[0]["snr_db"],
        "dB"
    )

    print(
        "Label:",
        test_dataset[0]["label"]
    )


    # --------------------------------------------------------
    # Plot one sample
    # --------------------------------------------------------

    sample = test_dataset[11]

    plt.figure(figsize=(10, 4))

    plt.plot(
        sample["signal"]
    )

    plt.title(
        f"Sample {sample['sample_id']} | "
        f"SNR = {sample['snr_db']} dB | "
        f"Class = {sample['label']}"
    )

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.tight_layout()

    plt.show()
