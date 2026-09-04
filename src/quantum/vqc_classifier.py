"""
Variational Quantum Classifier (VQC) / Quantum Neural Network.
Implements Parameterized Quantum Circuits (PQCs) with entangling layers
for classifying weak radar and sonar returns in high-clutter environments.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import List, Optional, Tuple, Dict, Any
from .feature_maps import angle_feature_map, zz_feature_map, entangled_angle_feature_map

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VariationalQuantumClassifier:
    """
    High-Performance Variational Quantum Classifier (VQC / QNN).
    Supports Entangled Angle & ZZ feature maps, Parameterized Quantum Circuits (PQCs),
    and fast hybrid optimization via PyTorch backprop (when available) or PennyLane autograd.
    """

    def __init__(
        self,
        n_qubits: int = 10,
        n_layers: int = 3,
        feature_map_type: str = "entangled_angle",  # 'entangled_angle', 'angle', 'zz', 'angle_yz'
        device_name: str = "default.qubit",
        seed: int = 42,
        use_torch: bool = True
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = feature_map_type.lower()
        self.seed = seed
        self.device_name = device_name
        self.use_torch = use_torch and HAS_TORCH

        self.dev = qml.device(device_name, wires=n_qubits)
        self.wires = list(range(n_qubits))

        np.random.seed(seed)
        if self.use_torch:
            torch.manual_seed(seed)
            self.torch_weights = torch.nn.Parameter(
                torch.randn(n_layers, n_qubits, 3, dtype=torch.float64) * 0.15
            )
            self.fc_head = nn.Linear(n_qubits, 1, dtype=torch.float64)

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _torch_circuit(x, weights):
                if self.feature_map_type in ["entangled_angle", "entangled"]:
                    entangled_angle_feature_map(x, wires=self.wires, reps=1)
                elif self.feature_map_type == "zz":
                    zz_feature_map(x, wires=self.wires, reps=1)
                elif self.feature_map_type in ["angle_yz", "yz"]:
                    angle_feature_map(x, wires=self.wires, rotation="YZ")
                else:
                    angle_feature_map(x, wires=self.wires, rotation="Y")

                qml.StronglyEntanglingLayers(weights, wires=self.wires)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            self._torch_qnode = _torch_circuit
        else:
            self.weights = pnp.random.normal(0, 0.15, (n_layers, n_qubits, 3), requires_grad=True)
            self.bias = pnp.array(0.0, requires_grad=True)
            self._qnode = qml.QNode(self._quantum_circuit, self.dev, interface="autograd")

    def _quantum_circuit(self, x: np.ndarray, weights: np.ndarray):
        """
        Quantum Circuit: Feature Map Encoding -> Trainable Variational Ansatz -> Pauli-Z Expectation.
        """
        if self.feature_map_type in ["entangled_angle", "entangled"]:
            entangled_angle_feature_map(x, wires=self.wires, reps=1)
        elif self.feature_map_type == "zz":
            zz_feature_map(x, wires=self.wires, reps=1)
        elif self.feature_map_type in ["angle_yz", "yz"]:
            angle_feature_map(x, wires=self.wires, rotation="YZ")
        else:
            angle_feature_map(x, wires=self.wires, rotation="Y")

        qml.StronglyEntanglingLayers(weights, wires=self.wires)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: np.ndarray, weights: Optional[np.ndarray] = None, bias: Optional[float] = None) -> float:
        """Forward pass for a single input vector."""
        if self.use_torch:
            with torch.no_grad():
                x_t = torch.as_tensor(x, dtype=torch.float64)
                outs = torch.stack(self._torch_qnode(x_t, self.torch_weights))
                logits = self.fc_head(outs.unsqueeze(0)).squeeze()
                return float(logits.item())
        else:
            w = self.weights if weights is None else weights
            b = self.bias if bias is None else bias
            return float(self._qnode(x, w) + b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid probability P(y=1 | x) for batch of inputs.
        """
        if self.use_torch:
            with torch.no_grad():
                x_t = torch.as_tensor(X, dtype=torch.float64)
                q_outs = torch.stack([torch.stack(self._torch_qnode(xi, self.torch_weights)) for xi in x_t])
                logits = self.fc_head(q_outs).squeeze(-1)
                probs_1 = torch.sigmoid(logits).cpu().numpy()
        else:
            raw_outputs = np.array([self.forward(x, self.weights, self.bias) for x in X])
            probs_1 = 1.0 / (1.0 + np.exp(-2.5 * raw_outputs))

        probs_0 = 1.0 - probs_1
        return np.column_stack((probs_0, probs_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (1 = Target/Mine, 0 = Clutter/Rock)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(np.int64)

    def cost(self, weights: np.ndarray, bias: float, X: np.ndarray, y: np.ndarray) -> float:
        """
        Binary Cross-Entropy (BCE) cost function with numerical stability.
        """
        eps = 1e-7
        raw_outputs = pnp.array([self.forward(x, weights, bias) for x in X])
        probs = 1.0 / (1.0 + pnp.exp(-2.5 * raw_outputs))
        probs = pnp.clip(probs, eps, 1.0 - eps)
        return -pnp.mean(y * pnp.log(probs) + (1.0 - y) * pnp.log(1.0 - probs))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 0.04,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the quantum variational weights using fast PyTorch Adam (or PennyLane Adam).
        """
        history = {"loss": [], "accuracy": []}
        n_samples = len(X_train)

        if self.use_torch:
            optimizer = optim.Adam(list(self.fc_head.parameters()) + [self.torch_weights], lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            X_tensor = torch.as_tensor(X_train, dtype=torch.float64)
            y_tensor = torch.as_tensor(y_train, dtype=torch.float64)

            for epoch in range(epochs):
                permutation = torch.randperm(n_samples)
                epoch_loss = 0.0
                batches = 0

                for b in range(0, n_samples, batch_size):
                    batch_idx = permutation[b:b + batch_size]
                    X_b = X_tensor[batch_idx]
                    y_b = y_tensor[batch_idx]

                    optimizer.zero_grad()
                    q_outs = torch.stack([torch.stack(self._torch_qnode(xi, self.torch_weights)) for xi in X_b])
                    logits = self.fc_head(q_outs).squeeze(-1)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches += 1

                avg_loss = epoch_loss / max(batches, 1)
                train_preds = self.predict(X_train)
                train_acc = float(np.mean(train_preds == y_train))
                history["loss"].append(avg_loss)
                history["accuracy"].append(train_acc)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        else:
            opt = qml.AdamOptimizer(stepsize=lr)
            for epoch in range(epochs):
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                for b in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[b:b + batch_size]
                    y_batch = y_shuffled[b:b + batch_size]

                    (self.weights, self.bias), current_loss = opt.step_and_cost(
                        lambda w, b_: self.cost(w, b_, X_batch, y_batch),
                        self.weights,
                        self.bias
                    )

                train_preds = self.predict(X_train)
                train_acc = np.mean(train_preds == y_train)
                total_loss = self.cost(self.weights, self.bias, X_train, y_train)

                history["loss"].append(float(total_loss))
                history["accuracy"].append(float(train_acc))

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        return history


if HAS_TORCH:
    class TorchQuantumClassifier(nn.Module):
        """
        Differentiable Hybrid PyTorch Quantum Neural Network (QNN).
        Leverages PyTorch backpropagation for fast GPU/CPU execution.
        """
        def __init__(self, n_qubits: int = 10, n_layers: int = 3, feature_map_type: str = "entangled_angle"):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.feature_map_type = feature_map_type
            dev = qml.device("default.qubit", wires=n_qubits)
            wires = list(range(n_qubits))

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qnode_fn(x, weights):
                if feature_map_type == "entangled_angle":
                    entangled_angle_feature_map(x, wires=wires)
                elif feature_map_type == "zz":
                    zz_feature_map(x, wires=wires)
                else:
                    angle_feature_map(x, wires=wires)
                qml.StronglyEntanglingLayers(weights, wires=wires)
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self._qnode_fn = qnode_fn
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3, dtype=torch.float64) * 0.15)
            self.fc = nn.Linear(n_qubits, 1, dtype=torch.float64)

        def forward(self, x_batch):
            q_outs = [torch.stack(self._qnode_fn(xi, self.weights)) for xi in x_batch]
            q_tensor = torch.stack(q_outs)
            return self.fc(q_tensor).squeeze(-1)
else:
    TorchQuantumClassifier = None
