"""
Classical Radar/Sonar Signal Processing & Machine Learning Baselines.
Provides standard Matched Filters, Support Vector Machines, Random Forests,
and Multilayer Perceptrons for benchmarking against Quantum AI/ML methods.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.signal import correlate


class ClassicalMatchedFilter:
    """
    Standard Classical Radar/Sonar Matched Filter.
    Computes cross-correlation of received return with reference transmit waveform:
    y(t) = \int r(\tau) s^*(\tau - t) d\tau
    """

    def __init__(self, reference_chirp: np.ndarray):
        self.reference = reference_chirp
        self.ref_energy = np.sum(np.abs(reference_chirp) ** 2)

    def process(self, rx_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Cross-correlates received signal with reference chirp.
        Returns correlation output and peak output SNR.
        """
        corr = correlate(rx_signal, self.reference, mode='same')
        peak_val = np.max(np.abs(corr))
        peak_snr = (peak_val ** 2) / (self.ref_energy + 1e-12)
        return corr, float(peak_snr)


class ClassicalSVM:
    """Classical Support Vector Machine with Radial Basis Function (RBF) Kernel."""

    def __init__(self, C: float = 1.0, gamma: str = 'scale'):
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)


class ClassicalRandomForest:
    """Classical Random Forest Classifier for acoustic/radar signatures."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)


def train_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate classical SVM, Random Forest, and MLP baselines.
    """
    baselines = {
        "Classical_SVM_RBF": ClassicalSVM(C=1.0),
        "Classical_RandomForest": ClassicalRandomForest(n_estimators=100),
        "Classical_MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
    }

    results = {}
    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        acc = np.mean(preds == y_test)
        results[name] = {
            "model": clf,
            "accuracy": float(acc),
            "predictions": preds,
            "probabilities": probs
        }

    return results
