"""
Quantum Feature Maps & State Encodings for Radar and Sonar Signals.
Maps classical time-series, I/Q samples, and frequency attributes into Hilbert space.
"""

import numpy as np
import pennylane as qml
from typing import List, Optional


def angle_feature_map(x: np.ndarray, wires: List[int], rotation: str = "Y"):
    """
    Angle Embedding: Encodes feature vector x as single-qubit rotations.
    |x⟩ = ⊗_{i=1}^N R_gate(x_i) |0⟩
    """
    for i, wire in enumerate(wires):
        val = x[i] if i < len(x) else 0.0
        if rotation.upper() == "Y":
            qml.RY(val, wires=wire)
        elif rotation.upper() == "Z":
            qml.RZ(val, wires=wire)
        elif rotation.upper() == "X":
            qml.RX(val, wires=wire)
        else:
            qml.RY(val, wires=wire)


def zz_feature_map(x: np.ndarray, wires: List[int], reps: int = 2):
    """
    ZZ-Feature Map (Entangled Non-linear Quantum Feature Map).
    Introduces non-linear multi-qubit correlations to separate cluttered radar/sonar returns:
    U_{\Phi(x)} = \prod ( e^{i \sum_{i,j} (\pi - x_i)(\pi - x_j) Z_i Z_j} \cdot \prod_i H_i R_z(2 x_i) H_i )
    """
    n_qubits = len(wires)
    for _ in range(reps):
        # 1. Hadamard layer
        for w in wires:
            qml.Hadamard(wires=w)
        
        # 2. Single-qubit RZ rotations
        for i, w in enumerate(wires):
            val = x[i] if i < len(x) else 0.0
            qml.RZ(2.0 * val, wires=w)
        
        # 3. Two-qubit entangling ZZ interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                xi = x[i] if i < len(x) else 0.0
                xj = x[j] if j < len(x) else 0.0
                phase = 2.0 * (np.pi - xi) * (np.pi - xj)
                
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(phase, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def amplitude_feature_map(x: np.ndarray, wires: List[int]):
    """
    Amplitude Embedding: Encodes 2^N classical normalized samples onto N qubits.
    |ψ_x⟩ = \sum_{k=0}^{2^N-1} x_k |k⟩
    """
    n_dim = 2 ** len(wires)
    x_padded = np.zeros(n_dim)
    length = min(len(x), n_dim)
    x_padded[:length] = x[:length]
    
    norm = np.linalg.norm(x_padded)
    if norm > 1e-9:
        x_padded = x_padded / norm
    else:
        x_padded[0] = 1.0

    qml.AmplitudeEmbedding(features=x_padded, wires=wires, normalize=True)


def create_quantum_circuit_diagram(n_qubits: int = 4, encoding: str = "zz") -> str:
    """
    Generates an ASCII circuit diagram representation for the quantum feature map.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    dummy_x = np.linspace(0.1, 2.5, n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        if encoding.lower() == "zz":
            zz_feature_map(x, wires=list(range(n_qubits)), reps=1)
        elif encoding.lower() == "angle":
            angle_feature_map(x, wires=list(range(n_qubits)))
        else:
            angle_feature_map(x, wires=list(range(n_qubits)))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qml.draw(circuit)(dummy_x)
