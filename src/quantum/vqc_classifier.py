"""
Variational Quantum Classifier (VQC) / Quantum Neural Network.
Implements Parameterized Quantum Circuits (PQCs) with entangling layers
for classifying weak radar and sonar returns in high-clutter environments.
"""

import numpy as np
import pennylane as qml
from typing import List, Optional, Tuple, Dict, Any
from .feature_maps import angle_feature_map, zz_feature_map


class VariationalQuantumClassifier:
    """
    Native PennyLane Variational Quantum Classifier (VQC).
    Uses a quantum feature map followed by trainable variational layers.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        feature_map_type: str = "angle",  # 'angle' or 'zz'
        device_name: str = "default.qubit",
        seed: int = 42
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = feature_map_type
        self.seed = seed

        # Initialize quantum simulator device
        self.dev = qml.device(device_name, wires=n_qubits)
        self.wires = list(range(n_qubits))

        # Weight initialization shape for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        np.random.seed(seed)
        self.weights = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3), requires_grad=True)
        self.bias = np.array(0.0, requires_grad=True)

        # Construct quantum node
        self._qnode = qml.QNode(self._quantum_circuit, self.dev, interface="autograd")

    def _quantum_circuit(self, x: np.ndarray, weights: np.ndarray):
        """
        Quantum Circuit: Feature Map Encoding -> Trainable Variational Ansatz -> Pauli-Z Expectation.
        """
        # 1. Quantum State Encoding
        if self.feature_map_type == "zz":
            zz_feature_map(x, wires=self.wires, reps=1)
        else:
            angle_feature_map(x, wires=self.wires, rotation="Y")

        # 2. Variational Entangling Ansatz
        qml.StronglyEntanglingLayers(weights, wires=self.wires)

        # 3. Measurement (Expectation value of Pauli-Z on primary qubit)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: np.ndarray, weights: np.ndarray, bias: float) -> float:
        """Forward pass for a single input vector."""
        return self._qnode(x, weights) + bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid probability P(y=1 | x) for batch of inputs.
        """
        raw_outputs = np.array([self.forward(x, self.weights, self.bias) for x in X])
        # Sigmoid activation: 1 / (1 + exp(-raw))
        probs_1 = 1.0 / (1.0 + np.exp(-raw_outputs))
        probs_0 = 1.0 - probs_1
        return np.column_stack((probs_0, probs_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (1 = Target/Mine, 0 = Clutter/Rock)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(np.int64)

    def cost(self, weights: np.ndarray, bias: float, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy / Mean Squared Error cost function."""
        predictions = [self.forward(x, weights, bias) for x in X]
        predictions = np.array(predictions)
        # Shift target binary {0, 1} to {-1, +1} for direct Pauli-Z alignment
        y_shifted = 2 * y - 1
        loss = np.mean((predictions - y_shifted) ** 2)
        return loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 35,
        batch_size: int = 16,
        lr: float = 0.08,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the quantum variational weights using PennyLane's AdamOptimizer.
        """
        opt = qml.AdamOptimizer(stepsize=lr)
        n_samples = len(X_train)
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            # Mini-batch permutation
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for b in range(0, n_samples, batch_size):
                X_batch = X_shuffled[b:b + batch_size]
                y_batch = y_shuffled[b:b + batch_size]

                # Optimization step
                (self.weights, self.bias), current_loss = opt.step_and_cost(
                    lambda w, b_: self.cost(w, b_, X_batch, y_batch),
                    self.weights,
                    self.bias
                )

            # Evaluate training metrics
            train_preds = self.predict(X_train)
            train_acc = np.mean(train_preds == y_train)
            total_loss = self.cost(self.weights, self.bias, X_train, y_train)

            history["loss"].append(float(total_loss))
            history["accuracy"].append(float(train_acc))

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        return history


# Hybrid PyTorch / PennyLane Wrapper for GPU/Deep Learning Pipelines
try:
    import torch
    import torch.nn as nn

    class TorchQuantumClassifier(nn.Module):
        """
        Hybrid PyTorch Quantum Neural Network (QNN).
        Wraps PennyLane QNode as a differentiable PyTorch Module.
        """
        def __init__(self, n_qubits: int = 6, n_layers: int = 3):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qnode_fn(inputs, weights):
                angle_feature_map(inputs, wires=list(range(n_qubits)))
                qml.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.q_layer = qml.qnn.TorchLayer(qnode_fn, weight_shapes)
            self.fc = nn.Linear(n_qubits, 1)

        def forward(self, x):
            q_out = self.q_layer(x)
            logits = self.fc(q_out)
            return logits

except ImportError:
    TorchQuantumClassifier = None
