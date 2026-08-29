# 🌊 Quantum Radar & Sonar Signal-Processing & Defense Situational Awareness

[![Open In Colab (Main Project)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/25A31A0356/UC086-Quantum-Weak-Signal/blob/main/notebooks/Quantum_Radar_Sonar_Colab.ipynb)
[![Open In Colab (Run on Real QPU)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/25A31A0356/UC086-Quantum-Weak-Signal/blob/main/notebooks/Run_In_The_Quantum_Computer.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-green.svg)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Quantum AI / ML & Quantum Computing for extracting weak radar/sonar returns from noisy environments and providing real-time Maritime Situational Awareness.**  
> **Target Domains:** Naval Defense, Coastal Surveillance, Subsurface Threat & Mine Countermeasures, Maritime Security.

---

## 🏗️ 7-Stage End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1 ["1. Data Ingestion & Kaggle Pipeline"]
        A1[Sonar Mines vs Rocks Dataset\nKaggle: 'tsaiteja2008'] --> D[Signal Preprocessor & PCA]
        A2[Statoil SAR Maritime Radar] --> D
        A3[Synthetic Low-SNR LFM Generator\nRayleigh Clutter + AWGN] --> D
    end

    subgraph S2 ["2. Quantum State Encoding"]
        D --> B1[Angle Embedding\nRY Pauli Rotations]
        D --> B2[Entangling ZZ-Feature Map\nNon-linear Clutter Separation]
    end

    subgraph S3 ["3. Quantum AI / ML Engine"]
        B1 & B2 --> C1[Variational Quantum Classifier\nStrongly Entangling Layers]
        B2 --> C2[Quantum Support Vector Machine\nHilbert Fidelity Kernel QSVC]
    end

    subgraph S4 ["4. Performance & Defense Metrics"]
        C1 & C2 --> E1[ROC Curve / Detection Rate Pd]
        C1 & C2 --> E2[False Alarm Rate Pfa & F1-Score]
    end

    subgraph S5 ["5. Tactical Threat Detection"]
        E1 & E2 --> F1[CFAR Adaptive Thresholding]
        F1 --> F2[Threat Score 0-100 & Target Classification]
    end

    subgraph S6 ["6. Situational Awareness HUD"]
        F2 --> G1[Active Polar PPI Radar Scope]
        F2 --> G2[Quantum Confidence Gauge]
        F2 --> G3[Threat Level: RED / AMBER / GREEN]
    end

    subgraph S7 ["7. Final Prototype / Live Demo"]
        G1 & G2 & G3 --> H1[Real-Time Naval Defense Decision Support HUD]
    end
```

---

## 🎯 Tactical Pipeline Stages

### 4. Performance & Defense Metrics
- **ROC Curves**: Detection Probability ($P_d$) vs. False Alarm Rate ($P_{fa}$).
- **Quantum Fidelity Kernels**: Gram matrix heatmap of state overlaps in Hilbert space ($K_{ij} = |\langle \psi_i | \psi_j \rangle|^2$).

### 5. Threat Detection
- **Target Classification**: Distinguishes submerged metallic naval mines from seafloor rocks.
- **Threat Score (0 - 100)**: Multi-factor composite combining quantum probability and measurement confidence.
- **Detection Threshold**: Constant False Alarm Rate (CFAR) adaptive baseline.
- **False-Alarm Control**: Tight probability control for high-clutter sea environments.

### 6. Situational Awareness
- **Target Status**: `HOSTILE`, `SUSPICIOUS`, `CLEAR`.
- **Confidence Score**: Quantum state measurement certainty percentage ($50\% - 99.9\%$).
- **Threat Level**: 🔴 **CRITICAL (RED)**, 🟡 **ELEVATED (AMBER)**, 🟢 **LOW (GREEN)**.
- **Alert & Command Dashboard**: Polar PPI Scope and automated tactical countermeasures feed.

### 7. Final Prototype Demo
- Real-time interactive ping simulation executing through the Parameterized Quantum Circuit (PQC) and rendering the Tactical Situational Awareness HUD.

---

## 🚀 1-Click Interactive Google Colab Notebooks

| Notebook | Focus | 1-Click Launch |
| :--- | :--- | :--- |
| **`Quantum_Radar_Sonar_Colab.ipynb`** | **Full 7-Stage Pipeline**: Kaggle Data, VQC, QSVC, Threat Detection & Situational Awareness HUD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/25A31A0356/UC086-Quantum-Weak-Signal/blob/main/notebooks/Quantum_Radar_Sonar_Colab.ipynb) |
| **`Run_In_The_Quantum_Computer.ipynb`** | **Real Physical QPU Deployment**: IBM Quantum Cloud, 1024 Shots & Error Mitigation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/25A31A0356/UC086-Quantum-Weak-Signal/blob/main/notebooks/Run_In_The_Quantum_Computer.ipynb) |

---

## 📜 License
MIT License. Developed for Quantum Information Science & Defense Signal Processing.
