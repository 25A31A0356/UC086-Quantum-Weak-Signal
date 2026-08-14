"""
UC-086: Quantum-Enhanced Weak Signal Detection
Classical Machine Learning Baselines & Radar/Sonar Benchmark Detectors

Implements:
1. Energy Detector (Radiometer / Power Threshold)
2. Support Vector Machine (RBF and Linear SVM)
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. Multi-Layer Perceptron (MLP Neural Network)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class EnergyDetector:
    """Classical Radar/Sonar Energy Threshold Detector."""
    def __init__(self, pfa_target: float = 0.05):
        self.pfa_target = pfa_target
        self.threshold = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Calculate energy for clutter/noise class (y == 0)
        clutter_energies = np.sum(X[y == 0] ** 2, axis=1)
        # Set threshold to achieve target false alarm rate
        self.threshold = np.percentile(clutter_energies, (1.0 - self.pfa_target) * 100)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        energies = np.sum(X ** 2, axis=1)
        return (energies > self.threshold).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        energies = np.sum(X ** 2, axis=1)
        # Sigmoid scaled pseudo-probability
        probs_1 = 1.0 / (1.0 + np.exp(-(energies - self.threshold)))
        probs_0 = 1.0 - probs_1
        return np.vstack([probs_0, probs_1]).T


def get_classical_models(random_state: int = 42) -> dict:
    """Return dictionary of all classical benchmark models."""
    models = {
        "Energy Detector": EnergyDetector(pfa_target=0.05),
        "Classical SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=random_state),
        "Classical SVM (Linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, random_state=random_state),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state)
    }
    return models


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate detector performance across all radar/sonar tactical metrics."""
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df))
    else:
        y_prob = y_pred.astype(float)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)  # Probability of Detection Pd
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate False Alarm Probability Pfa: FP / (FP + TN)
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    pfa = fp / (fp + tn + 1e-10)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5
        
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_pd": float(rec),  # Probability of Detection Pd
        "pfa": float(pfa),        # Probability of False Alarm Pfa
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }
