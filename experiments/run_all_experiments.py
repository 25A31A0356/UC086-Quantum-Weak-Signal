"""
UC-086: Quantum-Enhanced Weak Signal Detection
Comprehensive Benchmarking and Evaluation Engine

Runs extensive evaluation across:
1. Public Datasets: UCI Sonar, UCI Ionosphere, Maritime Hydrophone Acoustics.
2. Controlled SNR Degradation Sweep (-25 dB to +5 dB) to benchmark sub-noise weak signal detection.
3. Compares all Classical vs Quantum algorithms with loss mitigation.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import load_sonar_dataset, load_ionosphere_radar_dataset, load_maritime_acoustic_dataset
from signal_processing.signal_processor import inject_noise_at_snr
from classical_ml.classical_detectors import get_classical_models, evaluate_classifier
from quantum_ml.qsvm_detector import QuantumSupportVectorClassifier
from quantum_ml.vqc_detector import VariationalQuantumClassifier

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_all_models_on_dataset(X_train, X_test, y_train, y_test, dataset_name: str) -> list:
    """Evaluate both Classical and Quantum models on a dataset."""
    results = []
    
    # 1. Classical Models
    classical_models = get_classical_models()
    for name, model in classical_models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)
        metrics["Model"] = name
        metrics["Paradigm"] = "Classical"
        metrics["Dataset"] = dataset_name
        results.append(metrics)
        
    # 2. Quantum VQC
    vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2, max_iter=60)
    vqc.fit(X_train, y_train)
    vqc_metrics = vqc.evaluate(X_test, y_test)
    vqc_metrics["Model"] = "VQC (Quantum)"
    vqc_metrics["Paradigm"] = "Quantum"
    vqc_metrics["Dataset"] = dataset_name
    results.append(vqc_metrics)
    
    # 3. Quantum QSVM (with Loss Mitigation & Aligned Kernel)
    qsvm = QuantumSupportVectorClassifier(n_qubits=4, reps=2, gamma=0.3, C=15.0, alpha_alignment=0.15)
    qsvm.fit(X_train, y_train)
    qsvm_metrics = qsvm.evaluate(X_test, y_test)
    qsvm_metrics["Model"] = "QSVM (Loss-Mitigated)"
    qsvm_metrics["Paradigm"] = "Quantum"
    qsvm_metrics["Dataset"] = dataset_name
    results.append(qsvm_metrics)
    
    return results


def run_snr_sweep_benchmark(snr_levels: list = [-25, -20, -15, -10, -5, 0, 5]):
    """
    Run SNR stress-test sweep on active sonar signals to evaluate weak signal recovery below noise floor.
    """
    print("\n=======================================================")
    print("[*] Running SNR Degradation Sweep (-25 dB to +5 dB)...")
    print("=======================================================")
    
    X_train_clean, X_test_clean, y_train, y_test, _ = load_sonar_dataset()
    sweep_results = []
    
    for snr in tqdm(snr_levels, desc="Evaluating SNR levels"):
        # Inject controlled K-distribution sea noise into test features
        X_test_noisy = np.array([
            inject_noise_at_snr(x, snr_db=snr, clutter_type="k_distribution", noise_seed=42 + i)
            for i, x in enumerate(X_test_clean)
        ])
        
        # Evaluate Energy Detector
        from classical_ml.classical_detectors import EnergyDetector
        ed = EnergyDetector(pfa_target=0.05).fit(X_train_clean, y_train)
        ed_met = evaluate_classifier(ed, X_test_noisy, y_test)
        sweep_results.append({"SNR_dB": snr, "Model": "Energy Detector", "Accuracy": ed_met["accuracy"], "Pd": ed_met["recall_pd"], "Pfa": ed_met["pfa"], "AUC": ed_met["roc_auc"]})
        
        # Evaluate Classical SVM (RBF)
        from sklearn.svm import SVC
        svm = SVC(kernel="rbf", C=10.0, probability=True, random_state=42).fit(X_train_clean, y_train)
        svm_met = evaluate_classifier(svm, X_test_noisy, y_test)
        sweep_results.append({"SNR_dB": snr, "Model": "Classical SVM (RBF)", "Accuracy": svm_met["accuracy"], "Pd": svm_met["recall_pd"], "Pfa": svm_met["pfa"], "AUC": svm_met["roc_auc"]})
        
        # Evaluate Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_clean, y_train)
        rf_met = evaluate_classifier(rf, X_test_noisy, y_test)
        sweep_results.append({"SNR_dB": snr, "Model": "Random Forest", "Accuracy": rf_met["accuracy"], "Pd": rf_met["recall_pd"], "Pfa": rf_met["pfa"], "AUC": rf_met["roc_auc"]})
        
        # Evaluate Quantum QSVM
        qsvm = QuantumSupportVectorClassifier(n_qubits=4, reps=2, gamma=0.3, C=15.0, alpha_alignment=0.15).fit(X_train_clean, y_train)
        qsvm_met = qsvm.evaluate(X_test_noisy, y_test)
        sweep_results.append({"SNR_dB": snr, "Model": "QSVM (Quantum)", "Accuracy": qsvm_met["accuracy"], "Pd": qsvm_met["recall_pd"], "Pfa": qsvm_met["pfa"], "AUC": qsvm_met["roc_auc"]})
        
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(RESULTS_DIR, "snr_sweep_benchmark.csv"), index=False)
    return df_sweep


def main():
    print("==================================================================")
    print("  UC-086 QUANTUM WEAK SIGNAL DETECTION: COMPREHENSIVE BENCHMARK  ")
    print("==================================================================")
    
    all_dataset_results = []
    
    # 1. Benchmark on UCI Sonar Dataset
    print("\n[*] Benchmarking on UCI Sonar (Mines vs Rocks)...")
    X_tr_s, X_te_s, y_tr_s, y_te_s, meta_s = load_sonar_dataset()
    res_sonar = evaluate_all_models_on_dataset(X_tr_s, X_te_s, y_tr_s, y_te_s, "UCI Sonar")
    all_dataset_results.extend(res_sonar)
    
    # 2. Benchmark on UCI Ionosphere Radar Dataset
    print("\n[*] Benchmarking on UCI Ionosphere (Phased-Array Radar)...")
    X_tr_i, X_te_i, y_tr_i, y_te_i, meta_i = load_ionosphere_radar_dataset()
    res_radar = evaluate_all_models_on_dataset(X_tr_i, X_te_i, y_tr_i, y_te_i, "UCI Ionosphere Radar")
    all_dataset_results.extend(res_radar)
    
    # 3. Benchmark on Maritime Acoustic Hydrophone Dataset
    print("\n[*] Benchmarking on Maritime Hydrophone Acoustic Dataset...")
    X_tr_m, X_te_m, y_tr_m, y_te_m, meta_m = load_maritime_acoustic_dataset()
    res_maritime = evaluate_all_models_on_dataset(X_tr_m, X_te_m, y_tr_m, y_te_m, "Maritime Acoustics")
    all_dataset_results.extend(res_maritime)
    
    # Format and save overall dataset benchmark table
    df_all = pd.DataFrame(all_dataset_results)
    csv_path = os.path.join(RESULTS_DIR, "public_datasets_benchmark.csv")
    json_path = os.path.join(RESULTS_DIR, "public_datasets_benchmark.json")
    df_all.to_csv(csv_path, index=False)
    
    with open(json_path, 'w') as f:
        json.dump(all_dataset_results, f, indent=2)
        
    print("\n==================================================================")
    print("                    PUBLIC DATASETS BENCHMARK TABLE               ")
    print("==================================================================")
    print(df_all[["Dataset", "Model", "Paradigm", "accuracy", "recall_pd", "pfa", "roc_auc"]].to_string(index=False))
    
    # Run SNR sweep benchmark
    df_sweep = run_snr_sweep_benchmark()
    
    print(f"\n[+] Benchmark completed successfully. Saved outputs to:\n    - {csv_path}\n    - {json_path}")


if __name__ == "__main__":
    main()
