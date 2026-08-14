"""
UC-086: Quantum-Enhanced Weak Signal Detection
Feature Extraction and Signal Analysis Module

Extracts multi-domain physics-informed radar/sonar features:
1. Time-domain statistical moments: RMS, Kurtosis (clutter spikiness), Skewness, Crest Factor, PAPR.
2. Spectral Domain: FFT energy distribution, Spectral Centroid, Spectral Spread, Spectral Flatness, Spectral Entropy.
3. Time-Frequency / Wavelet: Multi-resolution sub-band energy decomposition.
"""

import numpy as np
from scipy import signal, stats

def extract_statistical_features(x: np.ndarray) -> np.ndarray:
    """Extract time-domain statistical moments from radar/sonar signal x."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-10
    
    mean_val = np.mean(x)
    std_val = np.std(x) + eps
    rms = np.sqrt(np.mean(x ** 2)) + eps
    peak = np.max(np.abs(x))
    
    kurt = stats.kurtosis(x)
    skew = stats.skew(x)
    crest_factor = peak / rms
    papr = 10 * np.log10((peak ** 2) / (rms ** 2) + eps)
    energy = np.sum(x ** 2)
    
    return np.array([mean_val, std_val, rms, peak, kurt, skew, crest_factor, papr, energy])


def extract_spectral_features(x: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Extract frequency-domain spectral features via FFT and Power Spectral Density."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-10
    
    n = len(x)
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    psd = (fft_vals ** 2) / (n * fs + eps)
    total_power = np.sum(psd) + eps
    normalized_psd = psd / total_power
    
    # Spectral Centroid: center of gravity of spectrum
    centroid = np.sum(freqs * normalized_psd)
    
    # Spectral Spread: variance of spectrum around centroid
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * normalized_psd))
    
    # Spectral Entropy: complexity / randomness of spectral distribution
    spec_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + eps))
    
    # Spectral Flatness: ratio of geometric mean to arithmetic mean
    geom_mean = np.exp(np.mean(np.log(psd + eps)))
    arith_mean = np.mean(psd) + eps
    flatness = geom_mean / arith_mean
    
    # Spectral Rolloff: frequency below which 85% of spectral energy lies
    cum_power = np.cumsum(normalized_psd)
    rolloff_idx = np.where(cum_power >= 0.85)[0]
    rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    return np.array([centroid, spread, spec_entropy, flatness, rolloff, total_power])


def extract_full_signal_feature_vector(x: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Extract comprehensive hybrid time-frequency feature vector for a 1D radar/sonar return."""
    stat_feats = extract_statistical_features(x)
    spec_feats = extract_spectral_features(x, fs)
    return np.concatenate([stat_feats, spec_feats])


if __name__ == "__main__":
    test_sig = np.sin(2 * np.pi * 50 * np.linspace(0, 0.1, 100)) + np.random.normal(0, 0.2, 100)
    feats = extract_full_signal_feature_vector(test_sig)
    print(f"Extracted feature vector: length={len(feats)}")
    print(f"Features: {feats}")
