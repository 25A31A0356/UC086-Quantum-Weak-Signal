# M3 Feature Engineering Handoff

## Input from M2

- `radar_signals.npy`
- `radar_metadata.csv`

Dataset:
- 1000 signals
- 1000 samples per signal
- Classes: 0 and 1
- SNR levels: 10, 5, 0, -5, -10 dB

## Feature Extraction

`feature_extraction.py` extracts:

### Time-domain features
- mean
- std
- variance
- max
- min
- rms
- peak_to_peak
- energy

### Frequency-domain features
- dominant_frequency
- dominant_magnitude
- spectral_energy

Output:

`radar_features.csv`

## Feature Engineering

Selected features:

- mean
- std
- max
- min
- rms
- peak_to_peak
- energy
- dominant_frequency
- dominant_magnitude

Label:

- `0` = noise-only
- `1` = signal + noise

Output:

`engineered_features.csv`

## Train/Test Split

- Training: 80%
- Testing: 20%
- Random state: 42
- Stratified by label

Files:

- `train_indices.npy`
- `test_indices.npy`

## Scaling

`StandardScaler` is fitted using training data only.

Scaler:

`feature_scaler.pkl`

## M4/M5 Requirement

M4 and M5 must use:

- the same selected features
- the same train indices
- the same test indices
- the same scaler

Do not regenerate or randomly split the dataset again.
