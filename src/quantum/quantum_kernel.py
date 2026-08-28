"""
Quantum Kernel Estimator and Quantum Support Vector Classifier (QSVC).
Calculates quantum state fidelity overlap in Hilbert space for separating
weak radar/sonar returns from complex ocean/sea clutter.
"""

import numpy as np
import pennylane as qml
from typing import Optional, List
from sklearn.svm import SVC
from .feature_maps import angle_feature_map, zz_feature_map


class QuantumKernelEstimator:
    """
    Computes Quantum Kernel Gram Matrix:
    K(x_1, x_2) = |⟨0| U^†(x_2) U(x_1) |0⟩|^2
    """

    def __init__(
        self,
        n_qubits: int = 6,
        feature_map_type: str = "zz",  # 'zz' or 'angle'
        device_name: str = "default.qubit"
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.wires = list(range(n_qubits))
        self.dev = qml.device(device_name, wires=n_qubits)

        self._kernel_qnode = qml.QNode(self._kernel_circuit, self.dev)

    def _kernel_circuit(self, x1: np.ndarray, x2: np.ndarray):
        """
        Circuit: State Preparation U(x1) -> Adjoint State Preparation U^†(x2) -> Measurement of |0...0⟩
        """
        if self.feature_map_type == "zz":
            # Apply U(x1)
            zz_feature_map(x1, wires=self.wires, reps=1)
            # Apply U^†(x2) using PennyLane adjoint
            qml.adjoint(zz_feature_map)(x2, wires=self.wires, reps=1)
        else:
            angle_feature_map(x1, wires=self.wires, rotation="Y")
            qml.adjoint(angle_feature_map)(x2, wires=self.wires, rotation="Y")

        # Probability of state |00...0⟩ is the fidelity overlap
        return qml.probs(wires=self.wires)

    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute fidelity K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|^2."""
        probs = self._kernel_qnode(x1, x2)
        return float(probs[0])  # State index 0 is |00...0⟩

    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes full Gram matrix between datasets X1 and X2.
        If X2 is None, computes symmetric square matrix K(X1, X1).
        """
        n1 = len(X1)
        if X2 is None:
            # Symmetric matrix calculation
            K = np.ones((n1, n1), dtype=np.float64)
            for i in range(n1):
                for j in range(i + 1, n1):
                    val = self.compute_kernel_element(X1[i], X1[j])
                    K[i, j] = val
                    K[j, i] = val
            return K
        else:
            n2 = len(X2)
            K = np.zeros((n1, n2), dtype=np.float64)
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.compute_kernel_element(X1[i], X2[j])
            return K


class QuantumSVC:
    """
    Quantum Support Vector Classifier using a Quantum Kernel Matrix.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        feature_map_type: str = "zz",
        C: float = 1.0,
        device_name: str = "default.qubit"
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.C = C
        self.kernel_estimator = QuantumKernelEstimator(
            n_qubits=n_qubits,
            feature_map_type=feature_map_type,
            device_name=device_name
        )
        self.svc = SVC(kernel="precomputed", C=C, probability=True)
        self.X_train_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit QSVC using Quantum Kernel matrix on training data."""
        self.X_train_ = X_train
        K_train = self.kernel_estimator.compute_kernel_matrix(X_train)
        self.svc.fit(K_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict classes using Quantum Kernel test matrix."""
        K_test = self.kernel_estimator.compute_kernel_matrix(X_test, self.X_train_)
        return self.svc.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        K_test = self.kernel_estimator.compute_kernel_matrix(X_test, self.X_train_)
        return self.svc.predict_proba(K_test)
