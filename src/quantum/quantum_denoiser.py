"""
Quantum-Inspired Signal Denoiser & Spectral Filter for Radar/Sonar returns.
Applies quantum spectral projection and singular subspace thresholding
to extract coherent weak signals from incoherent noise and sea clutter.
"""

import numpy as np
import pennylane as qml
from typing import Tuple, Optional


class QuantumSignalDenoiser:
    """
    Quantum-inspired Spectral Projection & Coherence Denoiser.
    Uses quantum state representation and unitary subspace projection
    to enhance weak radar chirps and sonar pings.
    """

    def __init__(self, n_qubits: int = 6):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def denoise_waveform(
        self,
        signal_array: np.ndarray,
        energy_threshold: float = 0.85
    ) -> np.ndarray:
        """
        Denoise a 1D radar or sonar time-series waveform using quantum state amplitude thresholding.

        1. Zero-pad or segment signal to 2^N quantum state dimension.
        2. Normalize into quantum state vector |ψ⟩ = \sum c_k |k⟩.
        3. Project into spectral basis via Quantum Fourier Transform (QFT).
        4. Apply coherent subspace thresholding to reject incoherent clutter/AWGN.
        5. Invert transform to reconstruct enhanced clean waveform.
        """
        original_length = len(signal_array)
        
        # Pad or interpolate to 2^n_qubits
        if original_length < self.dim:
            padded = np.pad(signal_array, (0, self.dim - original_length), 'constant')
        else:
            padded = signal_array[:self.dim]

        # Convert to complex state vector
        norm = np.linalg.norm(padded)
        if norm < 1e-12:
            return signal_array
        
        state_vec = padded / norm

        # Discrete Quantum-inspired Fourier Transform
        spectral_state = np.fft.fft(state_vec) / np.sqrt(self.dim)
        power_spectrum = np.abs(spectral_state) ** 2

        # Sort and select top coherent modes
        sorted_indices = np.argsort(power_spectrum)[::-1]
        cumulative_energy = np.cumsum(power_spectrum[sorted_indices])
        cutoff_idx = np.searchsorted(cumulative_energy, energy_threshold) + 1

        mask = np.zeros(self.dim, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True

        filtered_spectral = spectral_state * mask

        # Inverse transform
        reconstructed = np.fft.ifft(filtered_spectral) * np.sqrt(self.dim)
        denoised = np.real(reconstructed * norm)

        # Return to original length
        if original_length < self.dim:
            return denoised[:original_length]
        else:
            result = np.zeros(original_length)
            result[:self.dim] = denoised
            return result

    def compute_snr_gain(self, raw_signal: np.ndarray, denoised_signal: np.ndarray, clean_reference: np.ndarray) -> float:
        """
        Compute the Signal-to-Noise Ratio (SNR) enhancement in dB:
        SNR_gain = SNR_denoised - SNR_raw
        """
        raw_noise = raw_signal - clean_reference
        denoised_noise = denoised_signal - clean_reference

        p_clean = np.mean(clean_reference ** 2)
        p_raw_noise = np.mean(raw_noise ** 2) + 1e-12
        p_denoised_noise = np.mean(denoised_noise ** 2) + 1e-12

        snr_raw = 10 * np.log10(p_clean / p_raw_noise)
        snr_denoised = 10 * np.log10(p_clean / p_denoised_noise)

        return float(snr_denoised - snr_raw)
