"""
Radar & Sonar Signal Simulator for Quantum Enhancement.
Simulates LFM radar chirps, active sonar pings, Doppler shifts,
Rayleigh/K-distributed sea clutter, and low-SNR noise environments.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class RadarSignalSimulator:
    """
    Simulates pulsed Linear Frequency Modulation (LFM / Chirp) Radar Returns
    with Doppler shift and compound sea clutter (Rayleigh/Weibull/AWGN).
    """

    def __init__(
        self,
        sampling_rate: float = 1e6,      # 1 MHz sampling
        pulse_duration: float = 1e-4,    # 100 microseconds pulse
        bandwidth: float = 2e5,          # 200 kHz sweep bandwidth
        carrier_freq: float = 1e7        # 10 MHz IF carrier
    ):
        self.fs = sampling_rate
        self.T = pulse_duration
        self.B = bandwidth
        self.fc = carrier_freq
        self.n_samples = int(self.fs * self.T)
        self.t = np.linspace(0, self.T, self.n_samples, endpoint=False)
        self.chirp_rate = self.B / self.T

    def generate_ideal_chirp(self, doppler_shift: float = 0.0, phase_offset: float = 0.0) -> np.ndarray:
        """
        Generate reference LFM transmit chirp: s(t) = exp(j * 2*pi * (fc*t + 0.5*K*t^2 + fd*t))
        """
        freq_inst = self.fc + doppler_shift + 0.5 * self.chirp_rate * self.t
        phase = 2 * np.pi * freq_inst * self.t + phase_offset
        return np.exp(1j * phase)

    def generate_noisy_radar_return(
        self,
        target_present: bool = True,
        snr_db: float = -10.0,
        clutter_to_noise_ratio_db: float = 10.0,
        doppler_shift: float = 1500.0,   # 1.5 kHz Doppler
        clutter_type: str = "rayleigh"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate composite received radar waveform: r(t) = s(t) + c(t) + n(t)

        Args:
            target_present: If True, injects weak target echo.
            snr_db: Signal-to-Noise Ratio (dB).
            clutter_to_noise_ratio_db: Clutter-to-Noise Ratio (CNR in dB).
            doppler_shift: Target Doppler velocity shift (Hz).
            clutter_type: 'rayleigh', 'k_distributed', or 'gaussian'.

        Returns:
            rx_signal (np.ndarray): Complex baseband received signal.
            time_axis (np.ndarray): Time vector.
        """
        # Noise (AWGN)
        noise_power = 1.0
        n_real = np.random.normal(0, np.sqrt(noise_power / 2), self.n_samples)
        n_imag = np.random.normal(0, np.sqrt(noise_power / 2), self.n_samples)
        noise = n_real + 1j * n_imag

        # Clutter (Sea Clutter model)
        cnr_linear = 10 ** (clutter_to_noise_ratio_db / 10.0)
        clutter_power = noise_power * cnr_linear

        if clutter_type == "rayleigh":
            # Rayleigh amplitude with uniform random phase
            c_amp = np.random.rayleigh(scale=np.sqrt(clutter_power / 2), size=self.n_samples)
            c_phase = np.random.uniform(0, 2 * np.pi, size=self.n_samples)
            clutter = c_amp * np.exp(1j * c_phase)
        elif clutter_type == "k_distributed":
            # Compound K-distribution (heavy-tailed spiky sea clutter)
            gamma_mod = np.random.gamma(shape=1.5, scale=1.0, size=self.n_samples)
            c_amp = np.sqrt(gamma_mod) * np.random.rayleigh(scale=np.sqrt(clutter_power / 2), size=self.n_samples)
            c_phase = np.random.uniform(0, 2 * np.pi, size=self.n_samples)
            clutter = c_amp * np.exp(1j * c_phase)
        else:
            # Standard complex Gaussian clutter
            clutter = np.random.normal(0, np.sqrt(clutter_power / 2), self.n_samples) + \
                      1j * np.random.normal(0, np.sqrt(clutter_power / 2), self.n_samples)

        # Target Signal
        if target_present:
            snr_linear = 10 ** (snr_db / 10.0)
            signal_power = noise_power * snr_linear
            signal_amplitude = np.sqrt(signal_power)
            target_signal = signal_amplitude * self.generate_ideal_chirp(doppler_shift=doppler_shift)
        else:
            target_signal = np.zeros(self.n_samples, dtype=complex)

        rx_signal = target_signal + clutter + noise
        return rx_signal, self.t


class SonarSignalSimulator:
    """
    Simulates Active Sonar Acoustic Pings & Echoes from Submerged Objects
    vs Natural Reverberant Ocean Floor Clutter.
    """

    def __init__(self, sampling_rate: float = 1e5, duration: float = 0.05):
        self.fs = sampling_rate
        self.duration = duration
        self.n_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

    def generate_sonar_ping(
        self,
        target_type: str = "metallic_mine",  # 'metallic_mine', 'rock', 'clutter_only'
        snr_db: float = -8.0
    ) -> np.ndarray:
        """
        Generate multi-harmonic resonant acoustic response.
        - Metallic mine: Sharp resonant frequencies + high specular return.
        - Rock / Seafloor: Diffuse, broadband acoustic scattering with damped resonance.
        """
        noise = np.random.normal(0, 1.0, self.n_samples)
        
        if target_type == "clutter_only":
            return noise

        signal = np.zeros(self.n_samples)
        snr_linear = 10 ** (snr_db / 20.0)

        if target_type == "metallic_mine":
            # Strong specular return + 3 elastic resonance harmonic frequencies
            frequencies = [3200.0, 5800.0, 8400.0]
            delays = [0.005, 0.008, 0.012]
            for f, d in zip(frequencies, delays):
                envelope = np.exp(-((self.t - d) ** 2) / (2 * (0.002 ** 2)))
                signal += envelope * np.sin(2 * np.pi * f * self.t)
        else:  # Rock / Seafloor
            # Diffuse scattering across broad low-Q modes
            frequencies = [2100.0, 4200.0]
            delays = [0.01, 0.02]
            for f, d in zip(frequencies, delays):
                envelope = np.exp(-((self.t - d) ** 2) / (2 * (0.006 ** 2)))
                signal += 0.6 * envelope * np.sin(2 * np.pi * f * self.t)

        return (signal * snr_linear) + noise


def generate_radar_clutter_dataset(
    n_samples: int = 400,
    n_qubits: int = 6,
    snr_db: float = -12.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes a full radar dataset at a specified SNR (dB) for quantum classifier benchmarking.
    
    Classes:
    - 1: Stealth target / Weak threat return present in heavy sea clutter.
    - 0: False alarm / Pure sea clutter and receiver noise return.
    """
    np.random.seed(random_state)
    radar = RadarSignalSimulator(sampling_rate=5e5, pulse_duration=1e-4)

    raw_features = []
    labels = []

    for i in range(n_samples):
        target_present = (i % 2 == 1)
        # Randomize Doppler and clutter dynamics per pulse
        doppler = np.random.uniform(800.0, 2500.0)
        rx, _ = radar.generate_noisy_radar_return(
            target_present=target_present,
            snr_db=snr_db,
            doppler_shift=doppler,
            clutter_type="k_distributed"
        )

        # Extract spectral energy bins via FFT (Radar Doppler-Range spectrum)
        fft_mag = np.abs(np.fft.fft(rx))[:len(rx)//2]
        raw_features.append(fft_mag)
        labels.append(1 if target_present else 0)

    X_raw = np.array(raw_features)
    y = np.array(labels)

    # Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # PCA down to n_qubits
    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    # Scale to [0, pi] for quantum angle embedding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train_pca)
    X_test = scaler.transform(X_test_pca)

    return X_train, X_test, y_train, y_test


def generate_sonar_pulse_dataset(
    n_samples: int = 300,
    n_qubits: int = 6,
    snr_db: float = -8.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes an active sonar pulse dataset (Submerged Metallic Mine vs Sea Rock).
    """
    np.random.seed(random_state)
    sonar = SonarSignalSimulator()

    raw_features = []
    labels = []

    for i in range(n_samples):
        is_mine = (i % 2 == 1)
        target_type = "metallic_mine" if is_mine else "rock"
        pulse = sonar.generate_sonar_ping(target_type=target_type, snr_db=snr_db)

        # Extract FFT frequency spectrum
        fft_mag = np.abs(np.fft.rfft(pulse))
        raw_features.append(fft_mag)
        labels.append(1 if is_mine else 0)

    X_raw = np.array(raw_features)
    y = np.array(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train_pca)
    X_test = scaler.transform(X_test_pca)

    return X_train, X_test, y_train, y_test
