"""
Unit & Integration Tests for Quantum Radar & Sonar Weak Signal Processing.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum.feature_maps import (
    angle_feature_map,
    entangled_angle_feature_map,
    zz_feature_map,
    angle_yz_feature_map,
    get_feature_map
)
from src.quantum.quantum_kernel import QuantumKernel, QuantumSVC
from src.quantum.vqc_classifier import VariationalQuantumClassifier
from src.data.synthetic_generator import generate_radar_clutter_dataset, generate_sonar_pulse_dataset
from src.data.sonar_loader import prepare_sonar_quantum_data
from src.evaluation.metrics import evaluate_detection_metrics


class TestQuantumPipeline(unittest.TestCase):

    def test_feature_maps_circuit(self):
        """Verify that all feature maps execute on PennyLane devices without error."""
        n_qubits = 4
        x = np.random.uniform(0, np.pi, n_qubits)
        wires = list(range(n_qubits))
        
        dev = qml.device("default.qubit", wires=n_qubits)
        
        for fm_type in ["angle", "entangled_angle", "zz", "angle_yz"]:
            fm_func = get_feature_map(fm_type)
            
            @qml.qnode(dev)
            def circuit(features):
                fm_func(features, wires=wires)
                return qml.state()
                
            state = circuit(x)
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 2 ** n_qubits)
            self.assertTrue(np.isclose(np.sum(np.abs(state) ** 2), 1.0))

    def test_quantum_kernel_properties(self):
        """Verify Quantum Kernel Gram matrix properties: K_ii = 1.0, K_ij in [0, 1], K = K^T."""
        n_qubits = 4
        X = np.random.uniform(0, np.pi, (8, n_qubits))
        
        kernel = QuantumKernel(n_qubits=n_qubits, feature_map_type="entangled_angle")
        K = kernel(X, X)
        
        self.assertEqual(K.shape, (8, 8))
        self.assertTrue(np.allclose(K, K.T, atol=1e-5), "Gram matrix must be symmetric")
        self.assertTrue(np.allclose(np.diag(K), 1.0, atol=1e-5), "Self-fidelity diagonal must be 1.0")
        self.assertTrue(np.all(K >= -1e-7) and np.all(K <= 1.0 + 1e-7), "Kernel entries must be in [0, 1]")

    def test_qsvc_classification(self):
        """Verify Quantum Support Vector Classifier training and prediction."""
        X_train, X_test, y_train, y_test = generate_radar_clutter_dataset(
            n_samples=40, n_qubits=4, snr_db=5.0, random_state=42
        )
        
        qsvc = QuantumSVC(n_qubits=4, feature_map_type="entangled_angle", C=4.0)
        qsvc.fit(X_train, y_train)
        
        preds = qsvc.predict(X_test)
        probs = qsvc.predict_proba(X_test)
        
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape, (len(y_test), 2))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        
        metrics = evaluate_detection_metrics(y_test, preds, probs[:, 1])
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)

    def test_vqc_classification(self):
        """Verify Variational Quantum Classifier (PyTorch accelerated) training and prediction."""
        X_train, X_test, y_train, y_test = generate_radar_clutter_dataset(
            n_samples=40, n_qubits=4, snr_db=5.0, random_state=42
        )
        
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2, feature_map_type="entangled_angle")
        history = vqc.fit(X_train, y_train, epochs=5, batch_size=16, lr=0.05, verbose=False)
        
        self.assertEqual(len(history["loss"]), 5)
        self.assertTrue(history["loss"][-1] <= history["loss"][0] + 0.5)
        
        preds = vqc.predict(X_test)
        probs = vqc.predict_proba(X_test)
        
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape, (len(y_test), 2))

    def test_synthetic_data_generators(self):
        """Verify synthetic radar and sonar data generators."""
        X_r_tr, X_r_te, y_r_tr, y_r_te = generate_radar_clutter_dataset(
            n_samples=50, n_qubits=6, snr_db=-10.0, random_state=42
        )
        self.assertEqual(X_r_tr.shape, (37, 6))
        self.assertEqual(X_r_te.shape, (13, 6))
        self.assertTrue(np.all((X_r_tr >= 0) & (X_r_tr <= np.pi)))
        
        X_s_tr, X_s_te, y_s_tr, y_s_te = generate_sonar_pulse_dataset(
            n_samples=50, n_qubits=6, snr_db=-10.0, random_state=42
        )
        self.assertEqual(X_s_tr.shape, (37, 6))
        self.assertEqual(X_s_te.shape, (13, 6))

    def test_sonar_loader(self):
        """Verify Sonar dataset loading and quantum preprocessing."""
        X_train, X_test, y_train, y_test, pca = prepare_sonar_quantum_data(n_qubits=6, random_state=42)
        self.assertEqual(X_train.shape[1], 6)
        self.assertEqual(X_test.shape[1], 6)
        self.assertTrue(np.all((X_train >= 0) & (X_train <= np.pi)))
        self.assertEqual(len(y_train) + len(y_test), 208)


if __name__ == "__main__":
    unittest.main()
