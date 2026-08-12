# Classical ML

This folder contains the classical machine learning baseline.
# M4 Classical Machine Learning

## Objective

Build classical machine-learning baselines for weak-signal classification using the features prepared by M3.

## Input

M4 uses the following M3 outputs:

- engineered_features.csv
- train_indices.npy
- test_indices.npy
- feature_scaler.pkl
- selected_features.json

## Models

Four classical ML models are evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. K-Nearest Neighbors (KNN)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Data Split

Training:
- 800 samples

Testing:
- 200 samples

The train/test indices are inherited from M3.

## Important

M4 does not regenerate the dataset, train/test split, or scaler.

The M3 scaler is reused to maintain consistency between classical and quantum ML experiments.

## Outputs

- results.csv
- best_classical_model.pkl
