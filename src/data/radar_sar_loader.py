"""
Synthetic Aperture Radar (SAR) Maritime Dataset Preprocessor.
Processes dual-polarization Sentinel-1 radar backscatter (HH / HV bands)
for Ship vs. Iceberg classification and coastal radar target extraction.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def load_sar_radar_dataset(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw SAR radar JSON data (e.g. train.json from Statoil dataset).
    
    Each sample contains:
    - band_1: 75x75 float array (HH polarization backscatter in dB)
    - band_2: 75x75 float array (HV polarization backscatter in dB)
    - inc_angle: Radar incidence angle
    - is_iceberg: Target label (1 = Iceberg, 0 = Ship)

    Returns:
        band_1 (np.ndarray): Shape (N, 75, 75)
        band_2 (np.ndarray): Shape (N, 75, 75)
        inc_angles (np.ndarray): Shape (N,)
        labels (np.ndarray): Shape (N,)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    # Process radar bands
    b1 = np.array([np.array(band).reshape(75, 75) for band in df['band_1']], dtype=np.float32)
    b2 = np.array([np.array(band).reshape(75, 75) for band in df['band_2']], dtype=np.float32)
    
    # Fill missing incidence angles with mean
    inc_angles = pd.to_numeric(df['inc_angle'], errors='coerce')
    inc_angles = inc_angles.fillna(inc_angles.mean()).values.astype(np.float32)
    
    labels = df['is_iceberg'].values.astype(np.int64)

    return b1, b2, inc_angles, labels


def extract_radar_features(
    band_1: np.ndarray,
    band_2: np.ndarray,
    inc_angles: np.ndarray
) -> np.ndarray:
    """
    Extract physical radar backscatter and statistical texture features
    from dual-pol SAR returns for quantum processing.
    
    Features extracted per target:
    - Peak radar cross section (RCS) in HH & HV
    - Mean and standard deviation backscatter (clutter statistics)
    - HH/HV polarization ratio (polarimetric signature)
    - Radar incidence angle
    - Radial energy distribution / spatial moment
    """
    N = band_1.shape[0]
    features = []

    for i in range(N):
        h = band_1[i]
        v = band_2[i]
        angle = inc_angles[i]

        # Statistical moments
        h_mean, h_std, h_max, h_min = np.mean(h), np.std(h), np.max(h), np.min(h)
        v_mean, v_std, v_max, v_min = np.mean(v), np.std(v), np.max(v), np.min(v)
        
        # Polarimetric ratio & contrast
        cross_ratio = (h_mean - v_mean)  # dB difference
        peak_contrast_h = h_max - h_mean
        peak_contrast_v = v_max - v_mean

        # Center patch target intensity (15x15 target bounding box)
        center_h = np.mean(h[30:45, 30:45])
        center_v = np.mean(v[30:45, 30:45])

        feat = [
            h_mean, h_std, h_max, h_min,
            v_mean, v_std, v_max, v_min,
            cross_ratio, peak_contrast_h, peak_contrast_v,
            center_h, center_v, angle
        ]
        features.append(feat)

    return np.array(features, dtype=np.float64)


def prepare_sar_quantum_data(
    json_path: str,
    n_qubits: int = 6,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    End-to-end pipeline from SAR radar JSON to scaled quantum feature vectors.
    """
    b1, b2, inc_angles, y = load_sar_radar_dataset(json_path)
    X_raw = extract_radar_features(b1, b2, inc_angles)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Scale to [0, pi] for quantum angle embedding
    q_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = q_scaler.fit_transform(X_train_pca)
    X_test = q_scaler.transform(X_test_pca)

    return X_train, X_test, y_train, y_test, pca
