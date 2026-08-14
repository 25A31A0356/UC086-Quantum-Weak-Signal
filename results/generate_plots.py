"""
UC-086: Quantum-Enhanced Weak Signal Detection
Plot Generation and Results Visualization Suite

Generates publication-quality charts:
1. SNR vs Detection Accuracy & Pd curves (-25 dB to +5 dB).
2. ROC Curves (Receiver Operating Characteristics) at low SNR (-15 dB).
3. Confusion Matrices comparing Classical ML vs QSVM.
4. Public Dataset Performance Benchmark comparison bar chart.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import load_sonar_dataset, load_ionosphere_radar_dataset, load_maritime_acoustic_dataset
from signal_processing.signal_processor import inject_noise_at_snr
from quantum_ml.qsvm_detector import QuantumSupportVectorClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

def plot_snr_vs_accuracy():
    """Generate SNR vs Detection Accuracy and Pd comparison graph."""
    csv_path = os.path.join(RESULTS_DIR, "snr_sweep_benchmark.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from experiments.run_all_experiments import run_snr_sweep_benchmark
        df = run_snr_sweep_benchmark()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    models = df["Model"].unique()
    colors = {"QSVM (Quantum)": "#0066cc", "Classical SVM (RBF)": "#cc3300", "Random Forest": "#009933", "Energy Detector": "#737373"}
    markers = {"QSVM (Quantum)": "o", "Classical SVM (RBF)": "s", "Random Forest": "^", "Energy Detector": "d"}
    
    for m in models:
        sub = df[df["Model"] == m].sort_values("SNR_dB")
        c = colors.get(m, "#333333")
        marker = markers.get(m, "o")
        lw = 2.5 if "Quantum" in m else 1.8
        
        ax1.plot(sub["SNR_dB"], sub["Accuracy"] * 100, label=m, color=c, marker=marker, linewidth=lw, markersize=7)
        ax2.plot(sub["SNR_dB"], sub["Pd"] * 100, label=m, color=c, marker=marker, linewidth=lw, markersize=7)
        
    # Annotate Quantum Advantage Regime
    ax1.axvspan(-20, -5, color='#e6f2ff', alpha=0.5, label='Quantum Advantage Zone')
    ax2.axvspan(-20, -5, color='#e6f2ff', alpha=0.5, label='Quantum Advantage Zone')
    
    ax1.set_title("(a) Detection Accuracy vs Signal-to-Noise Ratio (SNR)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Detection Accuracy (%)", fontsize=11, fontweight='bold')
    ax1.set_ylim([35, 105])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True)
    
    ax2.set_title("(b) Probability of Detection ($P_d$) vs SNR", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Probability of Detection $P_d$ (%)", fontsize=11, fontweight='bold')
    ax2.set_ylim([35, 105])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "snr_vs_accuracy.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_roc_curves_low_snr():
    """Generate ROC curves at sub-noise floor SNR (-15 dB)."""
    X_train, X_test, y_train, y_test, _ = load_sonar_dataset()
    
    # Inject harsh -15 dB noise
    X_test_noisy = np.array([
        inject_noise_at_snr(x, snr_db=-15.0, clutter_type="k_distribution", noise_seed=200 + i)
        for i, x in enumerate(X_test)
    ])
    
    # Train Models
    svm_rbf = SVC(kernel="rbf", C=10.0, probability=True, random_state=42).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    qsvm = QuantumSupportVectorClassifier(n_qubits=4, reps=2, gamma=0.3, C=15.0, alpha_alignment=0.15).fit(X_train, y_train)
    
    prob_svm = svm_rbf.predict_proba(X_test_noisy)[:, 1]
    prob_rf = rf.predict_proba(X_test_noisy)[:, 1]
    prob_qsvm = qsvm.predict_proba(X_test_noisy)[:, 1]
    
    fpr_svm, tpr_svm, _ = roc_curve(y_test, prob_svm)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
    fpr_qsvm, tpr_qsvm, _ = roc_curve(y_test, prob_qsvm)
    
    auc_svm = auc(fpr_svm, tpr_svm)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_qsvm = auc(fpr_qsvm, tpr_qsvm)
    
    plt.figure(figsize=(7.5, 6), dpi=300)
    plt.plot(fpr_qsvm, tpr_qsvm, color="#0066cc", lw=3, label=f"QSVM (Quantum Kernel) [AUC = {auc_qsvm:.3f}]")
    plt.plot(fpr_rf, tpr_rf, color="#009933", lw=2, linestyle="--", label=f"Random Forest [AUC = {auc_rf:.3f}]")
    plt.plot(fpr_svm, tpr_svm, color="#cc3300", lw=2, linestyle="-.", label=f"Classical SVM (RBF) [AUC = {auc_svm:.3f}]")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle=":", label="Random Guess (AUC = 0.500)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Alarm Rate ($P_{fa}$)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Detection ($P_d$)", fontsize=12, fontweight='bold')
    plt.title("Receiver Operating Characteristic (ROC) at -15 dB SNR\n(Sub-Noise Floor Radar/Sonar Target Detection)", fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "roc_curves_low_snr.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_confusion_matrices():
    """Generate side-by-side Confusion Matrix comparison."""
    X_train, X_test, y_train, y_test, _ = load_sonar_dataset()
    X_test_noisy = np.array([
        inject_noise_at_snr(x, snr_db=-15.0, clutter_type="k_distribution", noise_seed=300 + i)
        for i, x in enumerate(X_test)
    ])
    
    svm = SVC(kernel="rbf", C=10.0, random_state=42).fit(X_train, y_train)
    qsvm = QuantumSupportVectorClassifier(n_qubits=4, reps=2, gamma=0.3, C=15.0, alpha_alignment=0.15).fit(X_train, y_train)
    
    pred_svm = svm.predict(X_test_noisy)
    pred_qsvm = qsvm.predict(X_test_noisy)
    
    cm_svm = confusion_matrix(y_test, pred_svm)
    cm_qsvm = confusion_matrix(y_test, pred_qsvm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    
    labels = ["Clutter / Rock", "Threat / Mine"]
    
    # Classical SVM
    im1 = ax1.imshow(cm_svm, cmap="Reds", alpha=0.8)
    ax1.set_title("Classical SVM (RBF) at -15 dB SNR", fontsize=12, fontweight='bold')
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels); ax1.set_yticklabels(labels)
    ax1.set_ylabel("True Tactical Label", fontweight='bold')
    ax1.set_xlabel("Predicted Label", fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm_svm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
            
    # QSVM
    im2 = ax2.imshow(cm_qsvm, cmap="Blues", alpha=0.8)
    ax2.set_title("Quantum QSVM at -15 dB SNR", fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels); ax2.set_yticklabels(labels)
    ax2.set_ylabel("True Tactical Label", fontweight='bold')
    ax2.set_xlabel("Predicted Label", fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_qsvm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
            
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_dataset_benchmark_bar_chart():
    """Generate public dataset benchmark comparison bar chart."""
    csv_path = os.path.join(RESULTS_DIR, "public_datasets_benchmark.csv")
    if not os.path.exists(csv_path):
        from experiments.run_all_experiments import main as run_exp
        run_exp()
        
    df = pd.read_csv(csv_path)
    
    selected_models = ["Classical SVM (RBF)", "Random Forest", "VQC (Quantum)", "QSVM (Loss-Mitigated)"]
    df_filtered = df[df["Model"].isin(selected_models)]
    
    pivot_acc = df_filtered.pivot(index="Dataset", columns="Model", values="accuracy") * 100
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    pivot_acc.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", width=0.75)
    ax.set_title("Detection Accuracy Across Public Radar & Sonar Databases", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Public Dataset Benchmark", fontsize=12, fontweight='bold')
    ax.set_ylim([60, 105])
    ax.set_xticklabels(pivot_acc.index, rotation=0, fontweight='bold')
    ax.legend(title="Algorithm", frameon=True, loc="lower right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numerical labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "dataset_benchmark_bar_chart.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def main():
    print("=== Generating Publication Visualizations & Charts ===")
    plot_snr_vs_accuracy()
    plot_roc_curves_low_snr()
    plot_confusion_matrices()
    plot_dataset_benchmark_bar_chart()
    print("[+] All plots successfully generated in results/ directory.")

if __name__ == "__main__":
    main()
