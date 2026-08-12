# M5 — Quantum Machine Learning
## UC-086 — Quantum Weak Signal Detection

This module implements the quantum machine learning stage of the UC-086 Quantum Weak Signal Detection prototype.

The objective is to compare a Variational Quantum Classifier (VQC) with classical machine learning models for radar weak-signal classification.

---

## 1. M5 Objective

The M5 module converts engineered radar features into a quantum-compatible representation and trains a 4-qubit Variational Quantum Classifier.

The quantum model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The results are compared with the classical ML baselines developed in M4.

---

## 2. Input Data

M5 uses the engineered features generated during M3.

The quantum model uses four selected features:
- mean
- std
- rms
- energy

The original M3 train/test split is preserved.

Dataset Split:
- Training: 800 samples
- Testing: 200 samples
- Total: 1000 samples

---

## 3. Quantum Data Preparation

The selected classical features are standardized using a StandardScaler.

The scaled features are then mapped to a quantum-compatible angle range.

Transformation:
quantum_angle = pi * tanh(scaled_feature)

This produces four quantum input values for the four-qubit circuit.

---

## 4. Quantum Architecture

The M5 prototype uses a 4-qubit Variational Quantum Circuit.

Configuration:
- Qubits: 4
- Variational layers: 2
- Feature encoding: RY rotations
- Trainable gates: RY + RZ
- Entanglement: CNOT
- Measurement: Pauli-Z expectation
- Quantum framework: PennyLane
- Simulator: default.qubit

Circuit:
M3 Engineered Features
        ↓
Feature Selection
        ↓
4 Quantum Features
        ↓
StandardScaler
        ↓
Quantum Angle Encoding
        ↓
4-Qubit Variational Circuit
        ↓
RY + RZ Trainable Gates
        ↓
CNOT Entanglement
        ↓
Pauli-Z Measurement
        ↓
Quantum Probability
        ↓
Classification

---

## 5. Training

The Variational Quantum Classifier is trained using the Adam optimizer.

Training Configuration:
- Optimizer: Adam
- Learning rate: 0.05
- Variational layers: 2
- Qubits: 4
- Initial training prototype: 200 samples
- Random seed: 42

The quantum circuit output is converted from the expectation-value range [-1, 1] into a probability range [0, 1].

Binary cross-entropy is used as the training loss.

---

## 6. Quantum Model Results

The recorded M5 quantum experiment produced:

| Metric | Score |
|---|---:|
| Accuracy | 0.475 |
| Precision | 0.476636 |
| Recall | 0.510000 |
| F1 Score | 0.492754 |
| ROC-AUC | 0.4783 |

These values represent the recorded M5 quantum experiment.

---

## 7. Classical Baseline

The M4 classical ML module evaluated:
1. Logistic Regression
2. SVM
3. Random Forest
4. KNN

Recorded results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| KNN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Variational Quantum Classifier | 0.475 | 0.476636 | 0.510 | 0.492754 | 0.4783 |

---

## 8. Classical vs Quantum Finding

The initial Variational Quantum Classifier did not outperform the classical ML baselines on the current dataset.

The classical models achieved perfect scores on the current test set, while the initial quantum model performed substantially lower.

This result is recorded as an experimental finding and is not interpreted as evidence that quantum machine learning is inherently inferior.

The current dataset is synthetic and relatively small, so additional validation is required before drawing conclusions about real-world radar weak-signal detection.

---

## 9. Important Experimental Note

The classical models achieving 1.0 across all reported metrics should be investigated for possible dataset simplicity or information leakage before using the results as evidence of generalization.

Future validation should consider:
- More realistic radar signals
- Higher noise levels
- More difficult signal-to-noise ratios
- Independent test datasets
- Cross-validation
- Noise robustness
- Additional quantum circuit architectures
- Hyperparameter optimization

---

## 10. Output Files

The M5 module can produce:

- quantum_results.csv
- classical_vs_quantum.csv
- quantum_weights.npy
- quantum_scaler.pkl
- X_train_quantum.npy
- X_test_quantum.npy
- y_train_quantum.npy
- y_test_quantum.npy

File descriptions:
- quantum_results.csv — Quantum model evaluation metrics
- classical_vs_quantum.csv — Combined classical and quantum comparison
- quantum_weights.npy — Trained variational circuit parameters
- quantum_scaler.pkl — Scaler used for quantum feature preparation
- X_train_quantum.npy — Quantum training features
- X_test_quantum.npy — Quantum testing features
- y_train_quantum.npy — Training labels
- y_test_quantum.npy — Testing labels

---

## 11. Reproducibility

The quantum model is implemented using PennyLane.

Install the dependency with:

pip install pennylane

Main implementation:
quantum_model.py

Quantum simulator:
qml.device("default.qubit", wires=4)

---

## 12. M5 Pipeline

M3 Engineered Features
        ↓
Feature Selection
        ↓
Select 4 Quantum Features
        ↓
StandardScaler
        ↓
Quantum Angle Encoding
        ↓
4-Qubit Variational Circuit
        ↓
RY + RZ Trainable Gates
        ↓
CNOT Entanglement
        ↓
Pauli-Z Measurement
        ↓
Quantum Probability
        ↓
Classification
        ↓
Evaluation
        ↓
Classical vs Quantum Comparison

---

## 13. M5 Status

| Task | Status |
|---|---|
| Quantum feature preparation | Complete |
| Quantum dataset generation | Complete |
| PennyLane setup | Complete |
| 4-qubit circuit | Complete |
| Variational layers | Complete |
| Quantum training | Complete |
| Quantum evaluation | Complete |
| Quantum results | Complete |
| Classical comparison | Complete |
| Results documentation | Complete |

M5 STATUS: COMPLETE

---

## 14. Next Stage

The next module will focus on final validation and integration of the classical and quantum pipelines.

Future work:
- Model validation
- Robustness testing
- Performance comparison
- Final prediction pipeline
- Visualization
- Prototype integration
- Final project reporting

---

## 15. Project Context

Use Case: UC-086 — Quantum Weak Signal Detection

Goal: Investigate whether quantum machine learning can provide useful classification capabilities for weak radar signals and compare its performance with classical machine learning approaches.

This implementation is a prototype research demonstration and should not be considered a production radar detection system.
