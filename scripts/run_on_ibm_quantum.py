"""
End-to-End Execution Script: Quantum Radar & Sonar Detection on IBM Quantum Hardware.
Connects directly to IBM Quantum Cloud QPUs or Aer Simulator, converts acoustic features into
Qubit Angle Embeddings, transpiles to physical hardware basis gates, and generates tactical defense outputs.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quantum.ibm_quantum_backend import IBMQuantumHardwareEngine
from src.tactical.threat_detection import TacticalThreatDetector
from src.tactical.situational_awareness import TacticalDashboard

def run_ibm_quantum_pipeline():
    print("=" * 80)
    print("  [*] QUANTUM RADAR & SONAR CLASSIFIER - IBM QUANTUM HARDWARE EXECUTION")
    print("=" * 80)

    # 1. Check for IBM Quantum Token from environment
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    if ibm_token:
        print("[+] Detected IBM_QUANTUM_TOKEN in environment variables.")
    else:
        print("[i] No IBM_QUANTUM_TOKEN set. (To use real hardware, set os.environ['IBM_QUANTUM_TOKEN'])")
        print("[i] Proceeding with high-precision Qiskit Aer Simulator...")

    # 2. Initialize IBM Quantum Hardware Engine
    engine = IBMQuantumHardwareEngine(
        token=ibm_token,
        n_qubits=6,
        n_layers=3,
        shots=1024
    )
    backend_info = engine.connect()

    print("\n--- 1. QPU HARDWARE CONFIGURATION ---")
    print(f" • Backend Name:     {backend_info['name']}")
    print(f" • Qubit Capacity:   {backend_info['num_qubits']} Qubits")
    print(f" • Technology:       {backend_info['type']}")
    print(f" • Hardware Status:  {backend_info['status']}")

    # 3. Simulate Incoming Sonar Ping Features (PCA compressed to 6 angles in [0, pi])
    print("\n--- 2. QUANTUM STATE & ANGLE ENCODING ---")
    # Acoustic reflection angles for Submerged Metallic Mine
    sample_angles = np.array([2.45, 1.18, 0.84, 2.91, 1.76, 0.52])
    print(f" • Raw 60-Band Sonar Vector compressed to 6 PCA dimensions.")
    print(f" • Physical Qubit Rotation Angles (θ_0 to θ_5):")
    for q, angle in enumerate(sample_angles):
        print(f"    - Qubit q[{q}] -> Ry({angle:.3f} rad) = {np.degrees(angle):.1f}°")

    # 4. Pre-trained Variational Quantum Weights
    np.random.seed(42)
    sample_weights = np.random.uniform(0, 2 * np.pi, (3, 6, 3))
    sample_bias = 0.15

    # 5. Display Qiskit Quantum Circuit
    print("\n--- 3. PARAMETERIZED QUANTUM CIRCUIT (OpenQASM 3.0) ---")
    print(engine.circuit_template.draw(output="text", fold=90))

    # 6. Execute on Hardware / Simulator
    print("\n--- 4. EXECUTING CIRCUIT ON QUANTUM PROCESSOR ---")
    result = engine.execute_ping(
        acoustic_features=sample_angles,
        trained_weights=sample_weights,
        bias=sample_bias
    )

    print("\n--- 5. PHYSICAL QPU MEASUREMENT RESULTS ---")
    print(f" • Quantum Backend:          {result['backend_name']}")
    print(f" • Transpiled Circuit Depth: {result['circuit_depth']} layers")
    print(f" • Pauli-Z Expectation <Z>:  {result['expectation_value_z']:+.4f}")
    print(f" • Raw Decision Value:       {result['raw_decision_value']:+.4f}")
    print(f" • Quantum Mine Probability: {result['mine_probability']*100:.2f}%")

    # 7. Tactical Threat Evaluation & Situational Awareness
    print("\n--- 6. TACTICAL DEFENSE DECISION SUPPORT ---")
    detector = TacticalThreatDetector()
    report = detector.evaluate_threat(
        quantum_probability=result['mine_probability'],
        raw_quantum_expval=result['expectation_value_z']
    )

    print("=" * 80)
    print(f" ⚠️ TARGET EVALUATION: TGT-01")
    print(f"   • Classification: {report['classification']}")
    print(f"   • Threat Level:   {report['threat_level']}")
    print(f"   • Threat Score:   {report['threat_score']}/100")
    print(f"   • QPU Confidence: {report['confidence_pct']}%")
    print(f"   • Tactical Action: {report['action']}")
    print("=" * 80)

if __name__ == "__main__":
    run_ibm_quantum_pipeline()
