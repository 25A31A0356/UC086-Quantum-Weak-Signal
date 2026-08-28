"""
Evaluation metrics & visualization utilities for Quantum Radar/Sonar Signal Processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)


def evaluate_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes critical defense radar/sonar metrics:
    - Probability of Detection (Pd / Recall)
    - Probability of False Alarm (Pfa / 1 - Specificity)
    - Accuracy, Precision, F1-Score, ROC-AUC
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pd_rate = tp / (tp + fn + 1e-12)  # Sensitivity / True Positive Rate
    pfa_rate = fp / (fp + tn + 1e-12) # False Alarm Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * (precision * pd_rate) / (precision + pd_rate + 1e-12)

    roc_auc_val = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = float(auc(fpr, tpr))

    return {
        "accuracy": float(accuracy),
        "Pd_detection_rate": float(pd_rate),
        "Pfa_false_alarm_rate": float(pfa_rate),
        "precision": float(precision),
        "f1_score": float(f1),
        "roc_auc": roc_auc_val,
        "confusion_matrix": cm.tolist()
    }


def compute_clutter_rejection_ratio(
    raw_signal_power: float,
    output_clutter_power: float
) -> float:
    """
    Computes Clutter Rejection Ratio (CRR in dB):
    CRR = 10 * log10(P_clutter_in / P_clutter_out)
    """
    crr = 10.0 * np.log10(raw_signal_power / (output_clutter_power + 1e-12))
    return float(crr)


def plot_roc_comparison(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None
):
    """
    Plot ROC Curves (Probability of Detection Pd vs False Alarm Pfa)
    comparing Quantum models (VQC, QSVC) with Classical baselines.
    
    models_dict: {"Model Name": (y_true, y_prob)}
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, lw=2.2)

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.2, label='Chance (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Probability of False Alarm ($P_{fa}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Detection ($P_d$)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Performance: Quantum vs Classical Signal Enhancement', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    return fig, ax


def run_snr_sweep_benchmark(
    snr_levels: List[float] = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0],
    n_samples: int = 150,
    n_qubits: int = 4
) -> Dict[str, List[float]]:
    """
    Evaluates detection accuracy across SNR levels (-20 dB to +5 dB)
    for Quantum VQC, Quantum SVC, and Classical SVM.
    """
    from ..data.synthetic_generator import generate_radar_clutter_dataset
    from ..quantum.vqc_classifier import VariationalQuantumClassifier
    from ..quantum.quantum_kernel import QuantumSVC
    from ..classical.baselines import ClassicalSVM

    results = {
        "snr_db": snr_levels,
        "Quantum_VQC": [],
        "Quantum_SVC": [],
        "Classical_SVM": []
    }

    for snr in snr_levels:
        X_train, X_test, y_train, y_test = generate_radar_clutter_dataset(
            n_samples=n_samples, n_qubits=n_qubits, snr_db=snr, random_state=42
        )

        # 1. Quantum VQC
        vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2)
        vqc.fit(X_train, y_train, epochs=15, batch_size=16, verbose=False)
        vqc_acc = np.mean(vqc.predict(X_test) == y_test)
        results["Quantum_VQC"].append(float(vqc_acc))

        # 2. Quantum SVC
        qsvc = QuantumSVC(n_qubits=n_qubits, feature_map_type="angle")
        qsvc.fit(X_train, y_train)
        qsvc_acc = np.mean(qsvc.predict(X_test) == y_test)
        results["Quantum_SVC"].append(float(qsvc_acc))

        # 3. Classical SVM
        svm = ClassicalSVM()
        svm.fit(X_train, y_train)
        svm_acc = np.mean(svm.predict(X_test) == y_test)
        results["Classical_SVM"].append(float(svm_acc))

    return results


def plot_snr_sweep_results(sweep_data: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot detection accuracy vs Signal-to-Noise Ratio (SNR).
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    snrs = sweep_data["snr_db"]

    for model_name, accs in sweep_data.items():
        if model_name == "snr_db":
            continue
        ax.plot(snrs, [a * 100 for a in accs], marker='o', lw=2.2, label=model_name)

    ax.set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Weak Signal Detection Robustness vs SNR", fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_radar_spectrogram(signal: np.ndarray, fs: float, title: str = "Radar Return Spectrogram"):
    """Plot time-frequency spectrogram of radar/sonar return."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    Pxx, freqs, bins, im = ax.specgram(signal, NFFT=128, Fs=fs, noverlap=64, cmap='viridis')
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)')
    plt.tight_layout()
    return fig, ax
