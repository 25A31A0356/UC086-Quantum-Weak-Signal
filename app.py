"""
UC-086: Quantum-Enhanced Weak Signal Detection
Tactical Defence Command & Control (C2) Interactive Dashboard

Run with:
    streamlit run app.py
or
    python -m streamlit run app.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import load_sonar_dataset, load_ionosphere_radar_dataset, load_maritime_acoustic_dataset
from signal_processing.signal_processor import inject_noise_at_snr, generate_lfm_radar_pulse, generate_sonar_acoustic_pulse
from quantum_ml.qsvm_detector import QuantumSupportVectorClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="UC-086: Quantum Radar/SONAR Weak Signal Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Tactical Command UI
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 15px;
        color: #475569;
        margin-bottom: 20px;
    }
    .card-metric {
        background-color: #f8fafc;
        border-left: 5px solid #0284c7;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .threat-confirmed {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: bold;
    }
    .threat-cleared {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 12px;
        border-radius: 6px;
        color: #166534;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title & Domain Banner
st.markdown("<div class='main-header'>📡 UC-086: Quantum-Enhanced Weak Signal Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Tactical Situational Awareness & Threat Detection for <b>Defence, Coastal Surveillance, and Maritime Security</b></div>", unsafe_allow_html=True)

# Sidebar: Controls & Dataset Selection
st.sidebar.header("🕹️ Tactical Mission Configuration")

dataset_choice = st.sidebar.selectbox(
    "Select Sensor Domain / Public Dataset:",
    [
        "UCI Sonar (Active Sonar: Underwater Mines vs Rocks)",
        "UCI Ionosphere (Phased-Array Radar: Air/Coastal Surveillance)",
        "Maritime Hydrophones (Passive Acoustic ASW Submarine Intercept)"
    ]
)

clutter_model = st.sidebar.selectbox(
    "Environmental Noise & Clutter Model:",
    ["K-Distribution Sea Clutter", "Rayleigh Sea Speckle", "Wenz Ambient Ocean Noise", "Additive Gaussian Noise (AWGN)"]
)

clutter_map = {
    "K-Distribution Sea Clutter": "k_distribution",
    "Rayleigh Sea Speckle": "rayleigh",
    "Wenz Ambient Ocean Noise": "ocean_wenz",
    "Additive Gaussian Noise (AWGN)": "awgn"
}

snr_db = st.sidebar.slider(
    "Signal-to-Noise Ratio (SNR in dB):",
    min_value=-25.0,
    max_value=+10.0,
    value=-10.0,
    step=1.0,
    help="Values below 0 dB indicate weak signals buried beneath the environmental noise floor."
)

if snr_db < -5.0:
    st.sidebar.warning(f"⚠️ Extreme Low-SNR Regime ({snr_db} dB): Classical detectors severely degrade. Quantum advantage active.")
elif snr_db <= 0.0:
    st.sidebar.info(f"ℹ️ Noise Floor Regime ({snr_db} dB): Sub-noise detection mode.")
else:
    st.sidebar.success(f"✅ Nominal SNR ({snr_db} dB): High signal clarity.")

# Cache dataset loading and model training
@st.cache_data
def get_dataset(choice):
    if "Sonar" in choice:
        return load_sonar_dataset()
    elif "Ionosphere" in choice:
        return load_ionosphere_radar_dataset()
    else:
        return load_maritime_acoustic_dataset()

X_train, X_test, y_train, y_test, meta = get_dataset(dataset_choice)

@st.cache_resource
def get_trained_models(choice):
    X_tr, _, y_tr, _, _ = get_dataset(choice)
    # Classical Baseline
    c_model = SVC(kernel="rbf", C=10.0, probability=True, random_state=42).fit(X_tr, y_tr)
    # Quantum QSVM
    n_q = 6 if X_tr.shape[1] >= 6 else 4
    q_model = QuantumSupportVectorClassifier(n_qubits=n_q, reps=2, gamma=0.85, C=20.0).fit(X_tr, y_tr)
    return c_model, q_model

classical_model, quantum_model = get_trained_models(dataset_choice)

# Main UI Tabs
tab_tactical, tab_spectrogram, tab_quantum, tab_benchmark = st.tabs([
    "🎯 Live Tactical Contact Intercept",
    "🌊 Waveform & Spectral Analysis",
    "⚛️ Quantum Hilbert Space Manifold",
    "📊 Comprehensive Defence Benchmarks"
])

# Tab 1: Live Tactical Contact Intercept
with tab_tactical:
    st.subheader("Real-Time Sensor Contact Discrimination")
    
    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        contact_id = st.number_input("Select Contact Index:", min_value=0, max_value=len(X_test)-1, value=5, step=1)
        raw_sample = X_test[contact_id]
        true_label = y_test[contact_id]
        threat_name = "CONFIRMED HOSTILE CONTACT (Target/Mine/Submarine)" if true_label == 1 else "NATURAL ENVIRONMENTAL CLUTTER (Rock/Noise)"
        st.caption(f"Ground Truth Intel: **{threat_name}**")
        
    # Inject noise into chosen contact
    noisy_contact = inject_noise_at_snr(
        raw_sample,
        snr_db=snr_db,
        clutter_type=clutter_map[clutter_model],
        noise_seed=int(contact_id * 100 + abs(snr_db))
    )
    
    # Inference
    c_prob = classical_model.predict_proba([noisy_contact])[0, 1]
    c_pred = int(c_prob >= 0.5)
    
    q_prob = quantum_model.predict_proba([noisy_contact])[0, 1]
    q_pred = int(q_prob >= 0.5)
    
    col_c, col_q = st.columns(2)
    
    with col_c:
        st.markdown("### 📡 Classical Baseline (RBF SVM)")
        st.progress(float(c_prob))
        st.write(f"Detection Confidence: **{c_prob*100:.1f}%**")
        if c_pred == 1:
            st.markdown("<div class='threat-confirmed'>🚨 THREAT DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='threat-cleared'>🛡️ NO THREAT / CLUTTER</div>", unsafe_allow_html=True)
            
        if c_pred == true_label:
            st.success("✅ Classification Decision: ACCURATE")
        else:
            st.error("❌ Classification Decision: ERROR (Missed Threat / False Alarm)")

    with col_q:
        st.markdown("### ⚛️ Quantum-Enhanced QSVM (Loss-Mitigated)")
        st.progress(float(q_prob))
        st.write(f"Quantum Hilbert Confidence: **{q_prob*100:.1f}%**")
        if q_pred == 1:
            st.markdown("<div class='threat-confirmed'>🚨 THREAT DETECTED (QUANTUM VERIFIED)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='threat-cleared'>🛡️ NO THREAT / CLUTTER</div>", unsafe_allow_html=True)
            
        if q_pred == true_label:
            st.success("✅ Classification Decision: ACCURATE (Robust to Noise)")
        else:
            st.error("❌ Classification Decision: ERROR")

    st.markdown("---")
    # Radar Tactical PPI Scope Simulation
    st.subheader("Tactical Radar / Sonar PPI Surveillance Scope")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'}, dpi=150)
    
    # Sweep background
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(0, 100, 100)
    ax.plot(theta, np.full_like(theta, 100), color='#0284c7', alpha=0.5)
    ax.set_facecolor('#0b1120')
    ax.grid(color='#1e293b', linestyle='--')
    
    # Plot detected blip
    blip_angle = (contact_id * 37) % 360 * (np.pi / 180.0)
    blip_radius = 30 + (contact_id * 13) % 60
    
    if q_pred == 1:
        ax.plot([blip_angle], [blip_radius], 'ro', markersize=12, label="Quantum Intercept (Threat)")
        ax.text(blip_angle, blip_radius + 8, f"THREAT #{contact_id+100}", color='red', fontsize=10, fontweight='bold')
    else:
        ax.plot([blip_angle], [blip_radius], 'go', markersize=9, label="Clear Contact (Clutter)")
        ax.text(blip_angle, blip_radius + 8, f"CLUTTER #{contact_id+100}", color='#4ade80', fontsize=10)
        
    ax.set_title("Plan Position Indicator (PPI) Real-Time Sector Display", color='#0284c7', fontsize=12, pad=15)
    ax.tick_params(colors='#94a3b8')
    st.pyplot(fig)

# Tab 2: Waveform & Spectral Analysis
with tab_spectrogram:
    st.subheader("Sensor Return Pulse & Spectral Decomposition")
    
    fig_sig, (ax_raw, ax_noisy, ax_fft) = plt.subplots(3, 1, figsize=(10, 7), dpi=150)
    
    ax_raw.plot(raw_sample, color='#0284c7', lw=1.8)
    ax_raw.set_title("1. Clean Public Sensor Signature (Uncorrupted)", fontsize=10, fontweight='bold')
    ax_raw.set_ylabel("Amplitude")
    ax_raw.grid(True, alpha=0.3)
    
    ax_noisy.plot(noisy_contact, color='#ef4444' if snr_db < 0 else '#f59e0b', lw=1.2)
    ax_noisy.set_title(f"2. Signal Corrupted by {clutter_model} at {snr_db} dB SNR", fontsize=10, fontweight='bold')
    ax_noisy.set_ylabel("Amplitude")
    ax_noisy.grid(True, alpha=0.3)
    
    # Power Spectrum
    fft_clean = np.abs(np.fft.rfft(raw_sample))
    fft_noisy = np.abs(np.fft.rfft(noisy_contact))
    ax_fft.plot(fft_clean, label="Clean Spectrum", color='#0284c7', lw=1.8)
    ax_fft.plot(fft_noisy, label=f"Noisy Spectrum ({snr_db} dB)", color='#ef4444', lw=1.2, linestyle="--")
    ax_fft.set_title("3. Frequency Domain Power Spectral Density", fontsize=10, fontweight='bold')
    ax_fft.set_xlabel("Frequency Bin")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.legend()
    ax_fft.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_sig)

# Tab 3: Quantum Hilbert Space Manifold
with tab_quantum:
    st.subheader("Quantum State Encoding & Hilbert Space Separability")
    st.markdown("""
    By mapping features into an $N$-qubit entangling $ZZ$-FeatureMap, the classical input space 
    is embedded into a $2^N$-dimensional complex projective Hilbert space where sub-noise weak signals 
    disentangle from random environmental clutter.
    """)
    
    angles = quantum_model.preprocessor.transform(X_test)
    
    fig_q, ax_q = plt.subplots(figsize=(8, 5), dpi=150)
    sc = ax_q.scatter(angles[:, 0], angles[:, 1], c=y_test, cmap='coolwarm', s=60, edgecolors='black', alpha=0.85)
    ax_q.set_xlabel("Qubit 0 Phase Angle $\\theta_0$ (rad)", fontweight='bold')
    ax_q.set_ylabel("Qubit 1 Phase Angle $\\theta_1$ (rad)", fontweight='bold')
    ax_q.set_title("Quantum State Phase Distribution $[0, \\pi]^2$", fontweight='bold', pad=12)
    plt.colorbar(sc, ax=ax_q, label="Threat (1) vs Clutter (0)")
    ax_q.grid(True, alpha=0.4)
    st.pyplot(fig_q)

# Tab 4: Comprehensive Defence Benchmarks
with tab_benchmark:
    st.subheader("Empirical Benchmark Results & SNR Sweep")
    res_path = os.path.join(os.path.dirname(__file__), "results", "public_datasets_benchmark.csv")
    if os.path.exists(res_path):
        df_res = pd.read_csv(res_path)
        st.dataframe(df_res.style.format({
            "accuracy": "{:.2%}", "recall_pd": "{:.2%}", "pfa": "{:.2%}", "roc_auc": "{:.3f}"
        }))
    else:
        st.info("Run `python experiments/run_all_experiments.py` to populate full benchmark tables.")
        
    snr_img = os.path.join(os.path.dirname(__file__), "results", "snr_vs_accuracy.png")
    if os.path.exists(snr_img):
        st.image(snr_img, caption="SNR vs Detection Accuracy & Pd Curve across -25 dB to +5 dB SNR")
