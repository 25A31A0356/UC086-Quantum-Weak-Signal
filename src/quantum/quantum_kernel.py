"""
Quantum Kernel Estimator and Quantum Support Vector Classifier (QSVC).
Calculates quantum state fidelity overlap in Hilbert space for separating
weak radar/sonar returns from complex ocean/sea clutter.
"""

import numpy as np
import pennylane as qml
from typing import Optional, List
from sklearn.svm import SVC
from .feature_maps import angle_feature_map, zz_feature_map, entangled_angle_feature_map


class QuantumKernelEstimator:
    """
    Computes Quantum Kernel Gram Matrix:
    K(x_1, x_2) = |⟨0| U^†(x_2) U(x_1) |0⟩|^2 = |⟨ψ(x_1) | ψ(x_2)⟩|^2
    """

    def __init__(
        self,
        n_qubits: int = 10,
        feature_map_type: str = "entangled_angle",  # 'entangled_angle', 'angle', 'zz'
        device_name: str = "default.qubit"
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type.lower()
        self.wires = list(range(n_qubits))
        self.dev = qml.device(device_name, wires=n_qubits)
        self.device_name = device_name

        # Statevector QNode for fast O(N) fidelity calculation
        self._state_qnode = qml.QNode(self._state_circuit, self.dev)
        # Pairwise adjoint QNode fallback
        self._kernel_qnode = qml.QNode(self._kernel_circuit, self.dev)

    def _apply_feature_map(self, x: np.ndarray):
        if self.feature_map_type in ["entangled_angle", "entangled"]:
            entangled_angle_feature_map(x, wires=self.wires, reps=1)
        elif self.feature_map_type == "zz":
            zz_feature_map(x, wires=self.wires, reps=1)
        elif self.feature_map_type in ["angle_yz", "yz"]:
            angle_feature_map(x, wires=self.wires, rotation="YZ")
        else:
            angle_feature_map(x, wires=self.wires, rotation="Y")

    def _state_circuit(self, x: np.ndarray):
        """Prepares quantum state |ψ(x)⟩."""
        self._apply_feature_map(x)
        return qml.state()

    def _kernel_circuit(self, x1: np.ndarray, x2: np.ndarray):
        """
        Circuit: State Preparation U(x1) -> Adjoint State Preparation U^†(x2) -> Measurement of |0...0⟩
        """
        self._apply_feature_map(x1)
        qml.adjoint(self._apply_feature_map)(x2)
        return qml.probs(wires=self.wires)

    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute fidelity K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|^2."""
        probs = self._kernel_qnode(x1, x2)
        return float(probs[0])  # State index 0 is |00...0⟩

    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes full Gram matrix between datasets X1 and X2.
        Uses fast vectorized statevector inner product when available on simulator device.
        """
        try:
            # Fast O(N) statevector computation: |<psi_i | psi_j>|^2
            states1 = np.array([self._state_qnode(x) for x in X1])
            if X2 is None:
                K = np.abs(states1 @ states1.conj().T) ** 2
            else:
                states2 = np.array([self._state_qnode(x) for x in X2])
                K = np.abs(states1 @ states2.conj().T) ** 2
            return np.clip(K, 0.0, 1.0)
        except Exception:
            # Fallback to pairwise circuit execution
            n1 = len(X1)
            if X2 is None:
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

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Calling estimator instance directly computes the Gram matrix."""
        return self.compute_kernel_matrix(X1, X2)


class QuantumSVC:
    """
    Quantum Support Vector Classifier using a Quantum Kernel Matrix.
    """

    def __init__(
        self,
        n_qubits: int = 10,
        feature_map_type: str = "entangled_angle",
        C: float = 8.0,
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
        self.svc = SVC(kernel="precomputed", C=C, probability=True, random_state=42)
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


# Aliases for flexible import
QuantumKernel = QuantumKernelEstimator

