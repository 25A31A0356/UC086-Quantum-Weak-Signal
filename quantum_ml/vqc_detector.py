"""
UC-086: Quantum-Enhanced Weak Signal Detection
Variational Quantum Classifier (VQC) Module

Implements a Parameterized Quantum Circuit (PQC) with:
1. Angle embedding feature layer.
2. Parameterized Variational Ansatz (RealAmplitudes / EfficientSU2 inspired).
3. Entangling CNOT layers.
4. Expectation measurement <Z_0> mapped to binary threat detection probabilities.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.quantum_preprocessor import QuantumFeaturePreprocessor

class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC) with loss mitigation and low-depth ansatz.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, max_iter: int = 80):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.n_params = n_qubits * (n_layers + 1)
        self.weights = np.random.uniform(-np.pi/4, np.pi/4, size=self.n_params)
        self.preprocessor = QuantumFeaturePreprocessor(n_qubits=n_qubits, angle_range=(0.0, np.pi))
        self.is_fitted = False
        
    def _apply_ry(self, psi: np.ndarray, theta: float, target: int, n: int) -> np.ndarray:
        """Apply single qubit Ry rotation."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        Ry = np.array([[c, -s], [s, c]], dtype=complex)
        
        psi_tensor = psi.reshape([2] * n)
        axes = [target] + [i for i in range(n) if i != target]
        psi_tensor = np.transpose(psi_tensor, axes)
        shape_rest = [2] * (n - 1)
        psi_mat = psi_tensor.reshape(2, -1)
        psi_mat = Ry @ psi_mat
        psi_tensor = psi_mat.reshape([2] + shape_rest)
        inv_axes = np.argsort(axes)
        psi_tensor = np.transpose(psi_tensor, inv_axes)
        return psi_tensor.reshape(-1)
        
    def _apply_cnot(self, psi: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
        """Apply 2-qubit CNOT gate."""
        idx = np.arange(2**n)
        ctrl_mask = 1 << (n - 1 - control)
        targ_mask = 1 << (n - 1 - target)
        
        # When control bit is 1, swap target bit state
        ctrl_is_one = (idx & ctrl_mask) != 0
        swapped_idx = np.where(ctrl_is_one, idx ^ targ_mask, idx)
        return psi[swapped_idx]

    def _forward_state(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Execute state preparation and parameterized variational ansatz."""
        n = self.n_qubits
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        
        # Feature Encoding: Ry(x[q])
        for q in range(n):
            psi = self._apply_ry(psi, x[q], q, n)
            
        param_idx = 0
        # Variational Layers
        for layer in range(self.n_layers):
            # Parameterized Ry rotations
            for q in range(n):
                psi = self._apply_ry(psi, weights[param_idx], q, n)
                param_idx += 1
            # Entangling CNOT ring / linear chain
            for q in range(n - 1):
                psi = self._apply_cnot(psi, q, q + 1, n)
                
        # Final rotation layer
        for q in range(n):
            psi = self._apply_ry(psi, weights[param_idx], q, n)
            param_idx += 1
            
        return psi

    def _measure_expectation_z0(self, psi: np.ndarray) -> float:
        """Compute expectation value <Z_0> on qubit 0: <psi| Z_0 |psi> in [-1, +1]."""
        n = self.n_qubits
        idx = np.arange(2**n)
        z0_eigenvals = np.where(((idx >> (n - 1)) & 1) == 0, 1.0, -1.0)
        prob = np.abs(psi) ** 2
        exp_val = np.sum(prob * z0_eigenvals)
        return float(np.real(exp_val))

    def _predict_prob_single(self, x: np.ndarray, weights: np.ndarray) -> float:
        """Map <Z_0> expectation to probability P(y=1) in [0, 1]."""
        psi = self._forward_state(x, weights)
        exp_z = self._measure_expectation_z0(psi)
        # Sigmoid or linear mapping: prob = (exp_z + 1.0) / 2.0
        prob = 1.0 / (1.0 + np.exp(-3.0 * exp_z))
        return prob

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train variational parameters via classical gradient-free COBYLA optimizer."""
        X_angles = self.preprocessor.fit_transform(X)
        
        # Subsample for faster robust convergence
        if len(X_angles) > 120:
            indices = np.random.choice(len(X_angles), size=120, replace=False)
            X_sub, y_sub = X_angles[indices], y[indices]
        else:
            X_sub, y_sub = X_angles, y
            
        def loss_fn(w):
            probs = np.array([self._predict_prob_single(x, w) for x in X_sub])
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            # Binary cross entropy loss
            bce = -np.mean(y_sub * np.log(probs) + (1 - y_sub) * np.log(1 - probs))
            return bce
            
        res = minimize(loss_fn, self.weights, method="COBYLA", options={"maxiter": self.max_iter})
        self.weights = res.x
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability distribution."""
        if not self.is_fitted:
            raise ValueError("VQC must be fitted before predicting.")
        X_angles = self.preprocessor.transform(X)
        p1 = np.array([self._predict_prob_single(x, self.weights) for x in X_angles])
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary detection labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate full detection performance metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
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
            "recall_pd": float(rec),
            "pfa": float(pfa),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "n_qubits": self.n_qubits
        }


if __name__ == "__main__":
    print("=== Testing Variational Quantum Classifier (VQC) ===")
    from data.data_loader import load_sonar_dataset
    X_tr, X_te, y_tr, y_te, meta = load_sonar_dataset()
    
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2, max_iter=50)
    print(f"Training VQC on {meta['dataset_name']}...")
    vqc.fit(X_tr, y_tr)
    
    metrics = vqc.evaluate(X_te, y_te)
    print(f"[+] VQC Evaluation Results:")
    print(f"    - Detection Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"    - Detection Probability (Pd): {metrics['recall_pd']*100:.2f}%")
    print(f"    - ROC-AUC: {metrics['roc_auc']:.3f}")
