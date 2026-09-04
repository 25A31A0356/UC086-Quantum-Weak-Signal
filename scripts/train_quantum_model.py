"""
Main Training & Benchmarking CLI for Quantum Radar/Sonar Signal Enhancement.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.sonar_loader import prepare_sonar_quantum_data
from src.data.synthetic_generator import generate_radar_clutter_dataset, generate_sonar_pulse_dataset
from src.quantum.vqc_classifier import VariationalQuantumClassifier
from src.quantum.quantum_kernel import QuantumSVC
from src.classical.baselines import train_classical_baselines
from src.evaluation.metrics import evaluate_detection_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Quantum AI/ML models on Radar/Sonar signal datasets.")
    parser.add_argument("--dataset", type=str, default="sonar", choices=["sonar", "radar_synthetic", "sonar_synthetic"],
                        help="Dataset to benchmark on")
    parser.add_argument("--qubits", type=int, default=10, help="Number of qubits in quantum register (default 10)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for VQC")
    parser.add_argument("--snr", type=float, default=-10.0, help="SNR (dB) if using synthetic radar/sonar generator")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for quantum optimizer (default 0.03)")
    parser.add_argument("--feature_map", type=str, default="entangled_angle", choices=["entangled_angle", "angle_yz", "angle", "zz"],
                        help="Quantum feature map encoding type")
    args = parser.parse_args()

    print("=" * 75)
    print("  QUANTUM RADAR & SONAR SIGNAL ENHANCEMENT - TRAINING & EVALUATION")
    print("=" * 75)
    print(f"[*] Dataset:       {args.dataset}")
    print(f"[*] Qubits:        {args.qubits}")
    print(f"[*] Feature Map:   {args.feature_map}")
    print(f"[*] VQC Epochs:    {args.epochs}")
    print(f"[*] Target SNR:    {args.snr} dB")

    # 1. Load / Prepare Dataset
    print("\n[+] Loading and preparing dataset...")
    if args.dataset == "sonar":
        X_train, X_test, y_train, y_test, pca = prepare_sonar_quantum_data(n_qubits=args.qubits)
        retained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"[OK] Retained Spectral Variance: {retained_var:.2f}% across {args.qubits} quantum channels")
    elif args.dataset == "radar_synthetic":
        X_train, X_test, y_train, y_test = generate_radar_clutter_dataset(
            n_samples=300, n_qubits=args.qubits, snr_db=args.snr
        )
    else:  # sonar_synthetic
        X_train, X_test, y_train, y_test = generate_sonar_pulse_dataset(
            n_samples=250, n_qubits=args.qubits, snr_db=args.snr
        )

    print(f"[OK] Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"[OK] Quantum Feature Dimension: {X_train.shape[1]}")

    # 2. Train Classical Baselines
    print("\n" + "-" * 50)
    print(" [1/3] Benchmarking Classical ML Baselines...")
    print("-" * 50)
    classical_results = train_classical_baselines(X_train, y_train, X_test, y_test)
    for model_name, res in classical_results.items():
        print(f"  -> {model_name:25s} | Accuracy: {res['accuracy']*100:.2f}%")

    # 3. Train Quantum Support Vector Classifier (QSVC)
    print("\n" + "-" * 50)
    print(f" [2/3] Training Quantum Support Vector Classifier (QSVC Hilbert Kernel)...")
    print("-" * 50)
    qsvc = QuantumSVC(n_qubits=args.qubits, feature_map_type=args.feature_map, C=8.0)
    qsvc.fit(X_train, y_train)
    
    qsvc_preds = qsvc.predict(X_test)
    qsvc_probs = qsvc.predict_proba(X_test)[:, 1]
    qsvc_metrics = evaluate_detection_metrics(y_test, qsvc_preds, qsvc_probs)

    print(f"  -> QSVC Test Accuracy:          {qsvc_metrics['accuracy']*100:.2f}%")
    print(f"  -> QSVC Detection Rate (Pd):    {qsvc_metrics['Pd_detection_rate']*100:.2f}%")
    print(f"  -> QSVC False Alarm Rate (Pfa): {qsvc_metrics['Pfa_false_alarm_rate']*100:.2f}%")
    print(f"  -> QSVC ROC-AUC:                {qsvc_metrics['roc_auc']:.4f}")

    # 4. Train Variational Quantum Classifier (VQC)
    print("\n" + "-" * 50)
    print(f" [3/3] Training Variational Quantum Classifier ({args.qubits} Qubits PQC)...")
    print("-" * 50)
    vqc = VariationalQuantumClassifier(n_qubits=args.qubits, n_layers=3, feature_map_type=args.feature_map)
    history = vqc.fit(X_train, y_train, epochs=args.epochs, batch_size=32, lr=args.lr, verbose=True)
    
    vqc_preds = vqc.predict(X_test)
    vqc_probs = vqc.predict_proba(X_test)[:, 1]
    vqc_metrics = evaluate_detection_metrics(y_test, vqc_preds, vqc_probs)

    print(f"\n  -> VQC Final Test Accuracy:     {vqc_metrics['accuracy']*100:.2f}%")
    print(f"  -> VQC Detection Rate (Pd):     {vqc_metrics['Pd_detection_rate']*100:.2f}%")
    print(f"  -> VQC False Alarm Rate (Pfa):  {vqc_metrics['Pfa_false_alarm_rate']*100:.2f}%")
    print(f"  -> VQC ROC-AUC:                 {vqc_metrics['roc_auc']:.4f}")

    print("\n" + "=" * 75)
    print("  SUMMARY BENCHMARK REPORT")
    print("=" * 75)
    print(f" {'Model':<30} | {'Accuracy':<10} | {'Detection (Pd)':<15} | {'ROC-AUC':<10}")
    print("-" * 75)
    for model_name, res in classical_results.items():
        print(f" {model_name:<30} | {res['accuracy']*100:>6.2f}%   | {'N/A':<15} | {'N/A':<10}")
    print(f" {'Quantum SVC (Hilbert Kernel)':<30} | {qsvc_metrics['accuracy']*100:>6.2f}%   | {qsvc_metrics['Pd_detection_rate']*100:>6.2f}%        | {qsvc_metrics['roc_auc']:>6.4f}")
    print(f" {'Quantum VQC (PQC QNN)':<30} | {vqc_metrics['accuracy']*100:>6.2f}%   | {vqc_metrics['Pd_detection_rate']*100:>6.2f}%        | {vqc_metrics['roc_auc']:>6.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
