# Features
# M3 — Feature Engineering

## UC-086: Quantum-Enhanced Weak Signal Detection in Noisy Radar/SONAR Signals

This directory contains the feature engineering pipeline for the UC-086 proof-of-concept.

The purpose of M3 is to transform extracted signal characteristics into a validated and reproducible feature dataset for the downstream classical and quantum/hybrid machine-learning models.

---

## 1. M3 Position in the Project

The overall project pipeline is:

Raw/Synthetic Signal
        ↓
Noise Addition
        ↓
Signal Processing
        ↓
Feature Extraction
        ↓
M3 Feature Engineering
        ↓
 ┌───────────────┐
 │               │
 ↓               ↓
M4              M5
Classical       Quantum/
ML              Hybrid ML
 │               │
 └───────┬───────┘
         ↓
   Experimental
    Comparison

---

## 2. M3 Responsibilities

M3 is responsible for:

- Validating the extracted feature dataset.
- Selecting the features used by downstream ML models.
- Creating a reproducible train/test split.
- Maintaining the same split for M4 and M5.
- Fitting the feature scaler using training data only.
- Saving preprocessing artifacts.
- Providing a documented handoff to M4 and M5.

---

## 3. Input

The current input file is:

```text
features/radar_features.csv
This folder contains preprocessing and feature extraction code.
