"""
IBM Quantum Hardware Execution Module.
Provides end-to-end integration with physical IBM Quantum processors (QPUs)
using Qiskit Runtime (EstimatorV2 / SamplerV2) and OpenQASM 3.0.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Qiskit Core & Runtime Imports
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Session
    _IBM_RUNTIME_AVAILABLE = True
except ImportError:
    _IBM_RUNTIME_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False


class IBMQuantumHardwareEngine:
    """
    Manages the complete lifecycle of executing radar/sonar quantum circuits
    on real IBM Quantum superconducting processors and local Aer simulators.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        instance: str = "ibm-q/open/main",
        backend_name: Optional[str] = None,
        n_qubits: int = 6,
        n_layers: int = 3,
        shots: int = 1024
    ):
        self.token = token
        self.instance = instance
        self.preferred_backend_name = backend_name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        self.service = None
        self.backend = None
        self.is_real_hardware = False
        self.backend_info: Dict[str, Any] = {}
        
        # Build parameterized Qiskit circuit template
        self.circuit_template, self.feature_params, self.weight_params = self._build_qiskit_circuit_template()
        self.observable = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])

    def _build_qiskit_circuit_template(self) -> Tuple[Any, Any, Any]:
        """
        Constructs the parameterized Qiskit QuantumCircuit for Angle Embedding + Entanglement + Variational Layers.
        """
        if not _QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed. Please install qiskit via `pip install qiskit`.")
        
        # 1. Feature parameters (input acoustic angles theta_0 .. theta_5)
        feature_params = ParameterVector("θ", self.n_qubits)
        
        # 2. Variational weights (3 angles per qubit per layer)
        n_weight_params = self.n_layers * self.n_qubits * 3
        weight_params = ParameterVector("φ", n_weight_params)
        
        qc = QuantumCircuit(self.n_qubits, name="QuantumRadarSonar_VQC")
        
        # --- Stage A: Quantum State Angle Embedding ---
        for i in range(self.n_qubits):
            qc.ry(feature_params[i], i)
            
        # Entangling ring across all qubits for non-linear correlation
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
            
        qc.barrier()
        
        # --- Stage B: Strongly Entangling Variational Layers ---
        param_idx = 0
        for l in range(self.n_layers):
            # Rotations on each qubit: Rz - Ry - Rz
            for q in range(self.n_qubits):
                qc.rz(weight_params[param_idx], q)
                qc.ry(weight_params[param_idx + 1], q)
                qc.rz(weight_params[param_idx + 2], q)
                param_idx += 3
                
            # Entanglement with periodic boundary conditions
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)
                
            qc.barrier()
            
        return qc, feature_params, weight_params

    def connect(self) -> Dict[str, Any]:
        """
        Connects to IBM Quantum Cloud or falls back gracefully to high-performance AerSimulator.
        """
        if self.token and _IBM_RUNTIME_AVAILABLE:
            try:
                print(f"[+] Authenticating with IBM Quantum Cloud (Instance: {self.instance})...")
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.token,
                    instance=self.instance
                )
                
                # Select specified or least busy operational QPU
                if self.preferred_backend_name:
                    self.backend = self.service.backend(self.preferred_backend_name)
                else:
                    self.backend = self.service.least_busy(simulator=False, operational=True)
                    
                self.is_real_hardware = True
                self.backend_info = {
                    "name": self.backend.name,
                    "num_qubits": self.backend.num_qubits,
                    "status": "OPERATIONAL (PHYSICAL QPU)",
                    "type": "Superconducting Transmon",
                    "cloud_provider": "IBM Quantum Platform"
                }
                print(f"[✓] Connected to Physical QPU: {self.backend.name} ({self.backend.num_qubits} Qubits)!")
                return self.backend_info
            except Exception as e:
                print(f"[!] IBM Cloud connection failed ({e}). Falling back to local Qiskit Aer Simulator.")
        
        # Fallback to local simulator
        if _AER_AVAILABLE:
            self.backend = AerSimulator()
            self.is_real_hardware = False
            self.backend_info = {
                "name": "AerSimulator (Local Statevector / QASM)",
                "num_qubits": self.n_qubits,
                "status": "SIMULATOR READY",
                "type": "Local Classical QPU Simulator",
                "cloud_provider": "Local Host"
            }
            print(f"[✓] Using Local Qiskit Aer Simulator ({self.n_qubits} Qubits).")
            return self.backend_info
        else:
            raise RuntimeError("Neither IBM Quantum Runtime nor Qiskit Aer is available.")

    def transpile_circuit_for_hardware(self, bound_circuit: Any) -> Any:
        """
        Transpiles the circuit into the physical coupling map and basis gates of the target QPU.
        """
        print(f"[+] Transpiling circuit for target backend: {self.backend_info.get('name', 'Backend')}...")
        transpiled_qc = transpile(
            bound_circuit,
            backend=self.backend,
            optimization_level=3,
            seed_transpiler=42
        )
        print(f"[✓] Transpilation complete! Gate Depth: {transpiled_qc.depth()} | Total Gates: {transpiled_qc.size()}")
        return transpiled_qc

    def execute_ping(
        self,
        acoustic_features: np.ndarray,
        trained_weights: np.ndarray,
        bias: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes a single acoustic radar/sonar ping through the Quantum Hardware.
        
        Args:
            acoustic_features: 6-element array of angle values in [0, pi]
            trained_weights: Flattened or (3, 6, 3) array of trained variational angles
            bias: Trained classical bias term
            
        Returns:
            Dictionary containing expectation value, mine probability, and QPU execution metadata.
        """
        # Flatten weights if passed as 3D array
        flat_weights = trained_weights.flatten()
        
        # Bind parameters to circuit
        param_dict = {}
        for i, val in enumerate(acoustic_features):
            param_dict[self.feature_params[i]] = float(val)
        for i, val in enumerate(flat_weights):
            param_dict[self.weight_params[i]] = float(val)
            
        bound_qc = self.circuit_template.assign_parameters(param_dict)
        
        # Hardware Transpilation
        transpiled_qc = self.transpile_circuit_for_hardware(bound_qc)
        
        # Execution via Qiskit Runtime Estimator or Aer Simulator
        print(f"[+] Submitting quantum execution job (Shots = {self.shots})...")
        
        if self.is_real_hardware and _IBM_RUNTIME_AVAILABLE:
            estimator = Estimator(mode=self.backend)
            job = estimator.run([(transpiled_qc, self.observable)])
            print(f"[+] IBM Quantum Job ID: {job.job_id()} | Status: Running on QPU...")
            result = job.result()
            exp_val = float(result[0].data.evs)
        else:
            # Local Aer measurement simulation
            from qiskit.primitives import StatevectorEstimator
            estimator = StatevectorEstimator()
            job = estimator.run([(transpiled_qc, self.observable)])
            result = job.result()
            exp_val = float(result[0].data.evs)
            
        # Compute Quantum Mine Probability via Sigmoid
        raw_val = exp_val + bias
        mine_prob = float(1.0 / (1.0 + np.exp(-raw_val)))
        
        return {
            "expectation_value_z": round(exp_val, 5),
            "raw_decision_value": round(raw_val, 5),
            "mine_probability": round(mine_prob, 4),
            "backend_name": self.backend_info.get("name"),
            "is_real_hardware": self.is_real_hardware,
            "circuit_depth": transpiled_qc.depth(),
            "circuit_qasm": transpiled_qc.qasm() if hasattr(transpiled_qc, "qasm") else "OpenQASM 3.0 Circuit"
        }
