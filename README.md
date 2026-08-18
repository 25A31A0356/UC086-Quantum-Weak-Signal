# UC086: Quantum-Enhanced Weak Signal Detection in Noisy Radar/SONAR Signals

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.3+-613394.svg)](https://qiskit.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-Defence%20%7C%20Coastal%20Surveillance%20%7C%20Maritime%20Security-navy.svg)]()
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/25A31A0356/UC086-Quantum-Weak-Signal)

A state-of-the-art framework for **Quantum-Enhanced Weak Signal Detection in Extreme Noise and Clutter Environments**, designed specifically for **Defence, Coastal Surveillance, and Maritime Security**.

---

## 🎯 Tactical Objectives & Outcomes
- **Enhanced Situational Awareness**: Recover weak, sub-noise floor return signals ($-15\text{ dB}$ to $-25\text{ dB}$ SNR) from stealth drones, silent diesel-electric submarines, and periscopes in severe sea clutter.
- **Quantum Advantage**: Up to **$+23.1\%$ gain in Detection Probability ($P_d$)** in deep sub-noise clutter where classical Energy Detectors and RBF SVMs fail.
- **Public Defense Datasets**: Fully automated ingestion of **UCI Sonar** (active mine countermeasure sonar), **UCI Ionosphere** (phased-array radar clutter), and **Maritime Hydrophone Acoustics**.
- **Quantum Loss Mitigation**: Employs low-depth $ZZ$-FeatureMaps and phase angle scaling $[0, \pi]$ to eliminate barren plateaus and minimize NISQ hardware error loss ($<1.8\%$).
- **Interactive Command & Control (C2) Dashboard**: Real-time tactical radar PPI sweep, sonar waterfall, SNR degradation slider, and live threat alert logger.

---

## 🏗️ Repository Architecture

```
UC086-Quantum-Weak-Signal/
├── data/                       # Automated Public Database Ingestion & Caching
│   ├── data_loader.py          # UCI Sonar, UCI Ionosphere, Maritime Acoustics Loader
│   └── README.md
├── signal_processing/          # Radar/SONAR Waveform & Clutter Simulation
│   ├── signal_processor.py     # LFM chirp, LFAS sonar, K-distribution clutter, AWGN, SNR injector
│   └── README.md
├── features/                   # Multi-Domain Feature Engineering & Quantum Encoding
│   ├── feature_extractor.py    # Statistical moments, FFT, Spectral Entropy, Wavelet energies
│   ├── quantum_preprocessor.py # PCA reduction & angle scaling [0, π] for multi-qubit mapping
│   └── README.md
├── classical_ml/               # Classical Baseline Detectors & Benchmarks
│   ├── classical_detectors.py  # CA-CFAR Energy Detector, RBF/Linear SVM, Random Forest, MLP
│   ├── train_classical.py      # Automated classical benchmark training
│   └── README.md
├── quantum_ml/                 # Quantum Machine Learning with Loss Mitigation
│   ├── quantum_kernels.py      # ZZ-FeatureMap & Pauli Quantum Kernel Estimator in 2^N Hilbert Space
│   ├── qsvm_detector.py        # Quantum Support Vector Machine (QSVM)
│   ├── vqc_detector.py         # Variational Quantum Classifier (VQC) with PQC ansatz
│   ├── quantum_loss_mitigation.py # Error mitigation & circuit fidelity analyzer
│   └── README.md
├── experiments/                # Rigorous Experimental Validation & Scenarios
│   ├── run_all_experiments.py  # Full SNR sweep (-25 to +5 dB) and multi-dataset benchmarks
│   ├── defense_scenarios.py    # 3 tactical operational defense missions
│   └── README.md
├── results/                    # Publication-Quality Charts, Tables & Evaluation
│   ├── generate_plots.py       # High-res ROC curves, SNR vs Accuracy, Confusion Matrices
│   ├── results_summary.md      # Comprehensive tabular results report
│   ├── snr_vs_accuracy.png     # Saved publication chart
│   ├── roc_curves_low_snr.png  # Saved ROC at -15 dB
│   ├── confusion_matrices.png  # Saved confusion matrix
│   └── dataset_benchmark_bar_chart.png
├── presentation/               # Tactical Briefing & Interactive CLI Demo
│   ├── defense_briefing.md     # Executive dossier for defense stakeholders
│   ├── run_demo.py             # Interactive standalone tactical terminal C2 demo
│   └── README.md
├── app.py                      # Interactive Web C2 Radar/SONAR Tactical Dashboard
└── requirements.txt            # System dependencies
```

---

## 📊 Benchmark Summary: Classical vs. Quantum

### 1. Performance on Public Databases

| Dataset / Mission Domain | Classical RBF SVM | Random Forest | VQC (Quantum) | QSVM (Loss-Mitigated) | Quantum Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UCI Sonar** *(Mine vs Rock)* | 86.54% | 84.62% | 84.62% | **92.31%** | **+5.77% Sensitivity Gain** |
| **UCI Ionosphere** *(Radar Clutter)* | 94.32% | 93.18% | 90.91% | **96.59%** | **+2.27% Gain** |
| **Maritime Acoustics** *(ASW Submarine)* | 88.00% | 89.33% | 86.67% | **94.67%** | **+5.34% Gain** |

### 2. Low-SNR Stress Test & Sub-Noise Target Detection ($P_d$)

| SNR (dB) | Energy Detector ($P_d$) | Classical SVM ($P_d$) | Quantum QSVM ($P_d$) | Operational Status |
| :---: | :---: | :---: | :---: | :--- |
| **+5 dB** | 92.3% | 96.2% | **100.0%** | Nominal tracking |
| **0 dB** | 73.1% | 88.5% | **96.2%** | Noise floor transition |
| **-10 dB** | 42.3% | 61.5% | **80.8%** | **+19.3% Quantum Gain** |
| **-15 dB** | 30.8% | 50.0% | **73.1%** | **+23.1% Quantum Advantage** |
| **-20 dB** | 19.2% | 38.5% | **61.5%** | Sub-noise floor target recovery |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Benchmarks & Generate Plots
```bash
# 1. Run all experiments and public dataset benchmarks
python experiments/run_all_experiments.py

# 2. Run tactical defence scenarios
python experiments/defense_scenarios.py

# 3. Generate high-resolution figures in results/
python results/generate_plots.py
```

### 3. Launch the Interactive Tactical C2 Web Dashboard
```bash
streamlit run app.py
```

### 4. Run Interactive Terminal Tactical Scanner
```bash
python presentation/run_demo.py
```

---

## 🛡️ Applications
- **Naval Anti-Submarine Warfare (ASW)**: Early detection of silent diesel-electric submarines buried in ambient ocean noise.
- **Coastal Border Surveillance**: Autonomous monitoring of littoral waters for semi-submersible narco craft and stealth speedboats in heavy surf clutter.
- **Air Defence & Counter-UAS**: Phased-array radar detection of low-RCS micro-drones under electronic counter-measures (ECM).

---
*Developed for UC-086 Quantum Weak Signal Detection Challenges.*
