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
        val = x[..., i]
        if rotation.upper() == "Y":
            qml.RY(val, wires=wire)
        elif rotation.upper() == "Z":
            qml.RZ(val, wires=wire)
        elif rotation.upper() == "X":
            qml.RX(val, wires=wire)
        elif rotation.upper() == "YZ" or rotation.upper() == "ZY":
            qml.RY(val, wires=wire)
            qml.RZ(val, wires=wire)
        else:
            qml.RY(val, wires=wire)


def angle_yz_feature_map(x: np.ndarray, wires: List[int]):
    """Angle embedding using dual-axis Y and Z rotations."""
    angle_feature_map(x, wires=wires, rotation="YZ")


def entangled_angle_feature_map(x: np.ndarray, wires: List[int], reps: int = 1):
    r"""
    Entangled Angle Feature Map:
    Combines single-qubit RY and RZ rotations with closed-ring CNOT entanglement.
    Extracts non-linear acoustic harmonics and inter-frequency correlation.
    """
    n_qubits = len(wires)
    for _ in range(reps):
        for i, w in enumerate(wires):
            val = x[..., i]
            qml.RY(val, wires=w)
        for i in range(n_qubits):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])
        for i, w in enumerate(wires):
            val = x[..., i]
            qml.RZ(val, wires=w)


def zz_feature_map(x: np.ndarray, wires: List[int], reps: int = 2):
    r"""
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
            val = x[..., i]
            qml.RZ(2.0 * val, wires=w)
        
        # 3. Two-qubit entangling ZZ interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                xi = x[..., i]
                xj = x[..., j]
                phase = 2.0 * (np.pi - xi) * (np.pi - xj)
                
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(phase, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def amplitude_feature_map(x: np.ndarray, wires: List[int]):
    r"""
    Amplitude Embedding: Encodes 2^N classical normalized samples onto N qubits.
    |\psi_x\rangle = \sum_{k=0}^{2^N-1} x_k |k\rangle
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


def get_feature_map(feature_map_type: str = "entangled_angle"):
    """
    Factory function retrieving the feature map function matching the string descriptor.
    """
    fmt = feature_map_type.lower()
    if fmt in ["entangled_angle", "entangled"]:
        return entangled_angle_feature_map
    elif fmt in ["angle_yz", "yz"]:
        return angle_yz_feature_map
    elif fmt in ["zz", "zz_feature_map"]:
        return zz_feature_map
    elif fmt in ["angle", "angle_feature_map"]:
        return angle_feature_map
    elif fmt in ["amplitude", "amplitude_embedding"]:
        return amplitude_feature_map
    else:
        return entangled_angle_feature_map

