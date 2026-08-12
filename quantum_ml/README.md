# Quantum ML

This folder contains the quantum/hybrid machine learning model.
# M5 Quantum Machine Learning

## Objective

Implement quantum machine-learning approaches for weak-signal classification.

## Input

M5 uses the feature-engineered dataset prepared by M3:

- engineered_features.csv
- train_indices.npy
- test_indices.npy
- feature_scaler.pkl
- selected_features.json

## Classical Baseline

M4 provides the classical ML baseline for comparison.

Models evaluated:

- Logistic Regression
- SVM
- Random Forest
- KNN

## Quantum ML

Quantum models will be evaluated against the M4 classical baseline.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
