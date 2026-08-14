"""
UC-086: Quantum-Enhanced Weak Signal Detection
Classical ML Training and Benchmarking Pipeline

Trains and evaluates all classical detectors on public radar and sonar datasets.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import load_sonar_dataset, load_ionosphere_radar_dataset, load_maritime_acoustic_dataset
from classical_ml.classical_detectors import get_classical_models, evaluate_classifier

def run_classical_benchmark_on_dataset(load_fn, dataset_label: str):
    """Run full classical benchmark suite on a specific dataset."""
    X_train, X_test, y_train, y_test, metadata = load_fn()
    print(f"\n==================================================")
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Domain: {metadata['domain']}")
    print(f"Samples: Train={len(X_train)}, Test={len(X_test)}, Features={X_train.shape[1]}")
    print(f"==================================================")
    
    models = get_classical_models()
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(f"[{name:<22}] Acc: {metrics['accuracy']*100:5.2f}% | Pd (Recall): {metrics['recall_pd']*100:5.2f}% | Pfa: {metrics['pfa']*100:5.2f}% | AUC: {metrics['roc_auc']:5.3f}")
        
    df_res = pd.DataFrame(results)[["Model", "accuracy", "recall_pd", "pfa", "f1_score", "roc_auc"]]
    return df_res

if __name__ == "__main__":
    print("=== Training Classical ML Baselines on Public Databases ===")
    res_sonar = run_classical_benchmark_on_dataset(load_sonar_dataset, "Sonar")
    res_radar = run_classical_benchmark_on_dataset(load_ionosphere_radar_dataset, "Ionosphere Radar")
    res_acoustics = run_classical_benchmark_on_dataset(load_maritime_acoustic_dataset, "Maritime Acoustics")
