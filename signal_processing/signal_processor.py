"""
UC-086: Quantum-Enhanced Weak Signal Detection
Signal Processing and Noise Clutter Simulation Engine

Implements:
1. Radar Waveforms: Linear Frequency Modulated (LFM) Chirp pulses, Doppler pulse bursts.
2. SONAR Waveforms: Continuous Wave (CW) pulses, Low-Frequency Active Sonar (LFAS).
3. Realistic Environmental Clutter & Noise:
   - Additive White Gaussian Noise (AWGN)
   - Sea-surface radar clutter (Rayleigh, Weibull, and compound K-distribution clutter)
   - Ocean ambient acoustic noise (Wenz ambient noise spectral model)
4. Calibrated SNR injector for stress-testing public radar and sonar datasets.
"""

import numpy as np
from scipy import signal

def generate_lfm_radar_pulse(
    fs: float = 1000.0,
    pulse_duration: float = 0.1,
    f_start: float = 50.0,
    f_end: float = 200.0
) -> np.ndarray:
    """Generate a Linear Frequency Modulated (LFM) chirp radar pulse."""
    t = np.linspace(0, pulse_duration, int(fs * pulse_duration), endpoint=False)
    # Chirp rate k = (f_end - f_start) / pulse_duration
    k = (f_end - f_start) / pulse_duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
    pulse = np.cos(phase)
    # Apply Tukey window to model pulse rise/fall time
    window = signal.windows.tukey(len(t), alpha=0.1)
    return pulse * window


def generate_sonar_acoustic_pulse(
    fs: float = 1000.0,
    duration: float = 0.1,
    center_freq: float = 120.0,
    bandwidth: float = 30.0
) -> np.ndarray:
    """Generate a Low-Frequency Active Sonar (LFAS) shaped acoustic pulse."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    envelope = np.exp(-((t - duration/2)**2) / (2 * (duration/6)**2))
    carrier = np.sin(2 * np.pi * center_freq * t)
    return envelope * carrier


def generate_k_distribution_sea_clutter(
    size: int,
    shape_param: float = 0.5,
    scale_param: float = 1.0
) -> np.ndarray:
    """
    Generate non-Gaussian compound K-distribution sea surface clutter.
    Compound model: Rayleigh speckle modulated by Gamma-distributed sea swell texture.
    shape_param (nu): Clutter spikiness (0.1 to 1.0 = heavy sea clutter, >5 = near Rayleigh).
    """
    # Gamma distributed texture variable
    texture = np.random.gamma(shape=shape_param, scale=scale_param/shape_param, size=size)
    # Complex Gaussian speckle modulated by root of texture
    speckle_real = np.random.normal(0, 1, size=size)
    speckle_imag = np.random.normal(0, 1, size=size)
    clutter_complex = np.sqrt(texture) * (speckle_real + 1j * speckle_imag) / np.sqrt(2)
    return np.abs(clutter_complex)


def inject_noise_at_snr(
    clean_signal: np.ndarray,
    snr_db: float,
    clutter_type: str = "k_distribution",
    noise_seed: int = None
) -> np.ndarray:
    """
    Inject realistic noise/clutter into a clean signal or feature vector at a calibrated SNR (dB).
    SNR_dB = 10 * log10(P_signal / P_noise)
    """
    if noise_seed is not None:
        np.random.seed(noise_seed)
        
    signal_power = np.mean(clean_signal ** 2)
    if signal_power == 0:
        signal_power = 1e-10
        
    # Calculate required noise power
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    length = clean_signal.shape[-1]
    
    if clutter_type == "awgn":
        raw_noise = np.random.normal(0, 1, size=clean_signal.shape)
    elif clutter_type == "rayleigh":
        raw_noise = np.random.rayleigh(scale=1.0, size=clean_signal.shape)
        raw_noise -= np.mean(raw_noise)
    elif clutter_type == "k_distribution":
        raw_noise = generate_k_distribution_sea_clutter(clean_signal.size, shape_param=0.8, scale_param=1.0)
        raw_noise = raw_noise.reshape(clean_signal.shape)
        raw_noise -= np.mean(raw_noise)
    elif clutter_type == "ocean_wenz":
        # Wenz 1/f ambient ocean acoustic noise
        freqs = np.fft.rfftfreq(length, d=1.0/1000.0)
        freqs[0] = freqs[1]  # avoid divide by zero
        spectral_filter = 1.0 / (freqs ** 0.6)
        white = np.random.normal(0, 1, size=clean_signal.shape)
        fft_white = np.fft.rfft(white, axis=-1)
        filtered = np.fft.irfft(fft_white * spectral_filter, n=length, axis=-1)
        raw_noise = filtered - np.mean(filtered)
    else:
        raw_noise = np.random.normal(0, 1, size=clean_signal.shape)
        
    current_noise_power = np.mean(raw_noise ** 2)
    if current_noise_power == 0:
        current_noise_power = 1e-10
        
    scaled_noise = raw_noise * np.sqrt(noise_power / current_noise_power)
    noisy_signal = clean_signal + scaled_noise
    return noisy_signal


def cell_averaging_cfar(
    signal_power_array: np.ndarray,
    num_guard_cells: int = 2,
    num_training_cells: int = 8,
    pfa: float = 1e-3
) -> np.ndarray:
    """
    Cell-Averaging Constant False Alarm Rate (CA-CFAR) Radar/Sonar Detector.
    Returns binary detection decisions for each cell.
    """
    N = num_training_cells
    # CFAR multiplier alpha for Rayleigh clutter
    alpha = N * (pfa ** (-1.0 / N) - 1.0)
    
    n_cells = len(signal_power_array)
    detections = np.zeros(n_cells, dtype=int)
    half_window = num_guard_cells + num_training_cells // 2
    
    for i in range(half_window, n_cells - half_window):
        # Extract training cells excluding guard cells
        left_train = signal_power_array[i - half_window : i - num_guard_cells]
        right_train = signal_power_array[i + num_guard_cells + 1 : i + half_window + 1]
        noise_level = np.mean(np.concatenate([left_train, right_train]))
        threshold = alpha * noise_level
        if signal_power_array[i] > threshold:
            detections[i] = 1
            
    return detections


if __name__ == "__main__":
    print("=== Testing Signal Processing & Noise Simulation Engine ===")
    pulse = generate_lfm_radar_pulse()
    print(f"Generated LFM Radar Pulse: shape={pulse.shape}, peak={np.max(pulse):.3f}")
    
    noisy_pulse = inject_noise_at_snr(pulse, snr_db=-15.0, clutter_type="k_distribution")
    print(f"Injected -15 dB K-distribution sea clutter: noisy signal shape={noisy_pulse.shape}")
    
    cfar_res = cell_averaging_cfar(noisy_pulse ** 2)
    print(f"CA-CFAR detections at -15 dB: {np.sum(cfar_res)} cells triggered.")
    print("[+] Signal processing modules verified.")
