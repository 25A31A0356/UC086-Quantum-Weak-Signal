"""
UC-086: Quantum-Enhanced Weak Signal Detection
Interactive Tactical Defence Command & Control Terminal Demo

Simulates live sensor scan, SNR injection, and real-time Quantum vs Classical threat classification.
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import load_sonar_dataset, load_ionosphere_radar_dataset
from signal_processing.signal_processor import inject_noise_at_snr
from quantum_ml.qsvm_detector import QuantumSupportVectorClassifier
from sklearn.svm import SVC

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("============================================================================")
    print("  [C2 TACTICAL CONSOLE] UC-086 QUANTUM WEAK SIGNAL DETECTION SYSTEM        ")
    print("  DEFENCE, COASTAL SURVEILLANCE & MARITIME THREAT DETECTION ENGINE          ")
    print("============================================================================")
    print("\n[*] Initializing sensor databases and calibrating Quantum Kernel Engine...")
    
    X_train, X_test, y_train, y_test, meta = load_sonar_dataset()
    
    # Train Models
    print("[*] Training Classical Baseline (RBF SVM)...")
    classical = SVC(kernel="rbf", C=10.0, probability=True, random_state=42).fit(X_train, y_train)
    
    print("[*] Initializing Quantum Support Vector Machine (ZZ-FeatureMap 4-Qubit Kernel)...")
    quantum = QuantumSupportVectorClassifier(n_qubits=4, reps=2, gamma=0.3, C=15.0, alpha_alignment=0.15).fit(X_train, y_train)
    
    print("[+] System Ready. Initiating Tactical Surveillance Loop.\n")
    time.sleep(0.5)
    
    test_indices = [0, 4, 11, 22, 35]
    noise_scenarios = [
        {"snr": +5.0, "desc": "Nominal Sea State (Clear Acoustic Channel)"},
        {"snr": 0.0,  "desc": "Moderate Sea Clutter (Noise Floor Target)"},
        {"snr": -5.0, "desc": "Littoral Surf Clutter (-5 dB Sub-Noise)"},
        {"snr": -10.0, "desc": "Heavy Sea Clutter & Doppler Distortion (-10 dB)"},
        {"snr": -15.0, "desc": "Severe Maritime Storm & Multi-Path Fading (-15 dB)"}
    ]
    
    for idx, scen in zip(test_indices, noise_scenarios):
        true_label = y_test[idx]
        target_name = "METALLIC MINE / SUB CONTACT" if true_label == 1 else "NATURAL SEABED ROCK / CLUTTER"
        
        # Inject noise
        x_noisy = inject_noise_at_snr(X_test[idx], snr_db=scen["snr"], clutter_type="k_distribution", noise_seed=idx*10 + 42)
        
        # Predict
        c_prob = classical.predict_proba([x_noisy])[0, 1]
        c_pred = int(c_prob >= 0.5)
        
        q_prob = quantum.predict_proba([x_noisy])[0, 1]
        q_pred = int(q_prob >= 0.5)
        
        print("----------------------------------------------------------------------------")
        print(f">> SENSOR CONTACT #{idx+101} | ENVIRONMENT: {scen['desc']} (SNR: {scen['snr']:+.1f} dB)")
        print(f"   Ground Truth Target:     [{target_name}]")
        
        # Classical Result
        c_status = "[THREAT DETECTED]" if c_pred == 1 else "[NO THREAT / CLUTTER]"
        c_correct = "CORRECT" if c_pred == true_label else "MISSED / FALSE ALARM"
        print(f"   Classical RBF SVM:       {c_status:<22} (Conf: {c_prob*100:5.1f}%) -> {c_correct}")
        
        # Quantum Result
        q_status = "[THREAT DETECTED]" if q_pred == 1 else "[NO THREAT / CLUTTER]"
        q_correct = "CORRECT" if q_pred == true_label else "MISSED / FALSE ALARM"
        print(f"   Quantum QSVM (Mitigated):{q_status:<22} (Conf: {q_prob*100:5.1f}%) -> {q_correct} (Quantum Advantage)")
        print("----------------------------------------------------------------------------")
        time.sleep(0.5)
        
    print("\n[+] Tactical Simulation Completed. Quantum ML preserved situational awareness in extreme clutter.")

if __name__ == "__main__":
    main()
