# Quantum ML

This folder contains the quantum/hybrid machine learning model.
# M5 — Quantum Machine Learning

## Objective

Develop a quantum machine-learning prototype for weak radar signal
classification and compare its performance with classical machine-learning
baselines.

## Input Data

The quantum model uses the feature-engineered dataset prepared during M3.

Input files:

- `engineered_features.csv`
- `train_indices.npy`
- `test_indices.npy`

## Quantum Feature Preparation

Four features were selected for the initial quantum prototype:

1. Mean
2. Standard deviation
3. RMS
4. Energy

These four features are mapped to four qubits.

The data-processing pipeline is:

M3 engineered features
→ feature selection
→ training/test split
→ quantum-specific scaling
→ angle transformation
→ quantum circuit

## Quantum Model

A Variational Quantum Classifier was implemented using PennyLane.

### Circuit

- Number of qubits: 4
- Number of variational layers: 2
- Feature encoding: RY rotations
- Trainable gates: Rotational gates
- Entanglement: CNOT gates
- Measurement: Pauli-Z expectation value

## Classical Baseline

The quantum model is compared against the M4 classical models:

- Logistic Regression
- SVM
- Random Forest
- KNN

## Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Experimental Results

Results are stored in:

- `quantum_results.csv`
- `classical_vs_quantum.csv`

The comparison should be interpreted as a prototype benchmark on the
current synthetic dataset.

## Important Limitation

The current radar dataset is synthetic and the classical models achieved
very high performance. Therefore, the results should not be interpreted
as evidence of real-world radar detection accuracy.

Further validation with realistic radar measurements and more difficult
signal-to-noise conditions is required.

## Output Files

- `quantum_data.py`
- `quantum_model.py`
- `quantum_results.csv`
- `quantum_weights.npy`
- `classical_vs_quantum.csv`
- `classical_vs_quantum_accuracy.png`
- `classical_vs_quantum_f1.png`
