"""
Tactical Threat Detection Module for Quantum Radar & Sonar Systems.
Computes Constant False Alarm Rate (CFAR) thresholds, threat scores, and target classification.
"""

import numpy as np


class TacticalThreatDetector:
    def __init__(self, cfar_pfa=0.01, threat_weight_prob=0.6, threat_weight_conf=0.4):
        """
        Initialize the Tactical Threat Detector.
        :param cfar_pfa: Desired Probability of False Alarm (e.g. 10^-2).
        :param threat_weight_prob: Weight for quantum class probability.
        :param threat_weight_conf: Weight for quantum fidelity confidence.
        """
        self.cfar_pfa = cfar_pfa
        self.w_prob = threat_weight_prob
        self.w_conf = threat_weight_conf

    def compute_cfar_threshold(self, noise_estimates):
        """
        Calculates Cell-Averaging CFAR adaptive threshold.
        T = alpha * P_noise
        """
        p_noise = np.mean(noise_estimates)
        alpha = len(noise_estimates) * (self.cfar_pfa ** (-1.0 / len(noise_estimates)) - 1.0)
        return alpha * p_noise

    def evaluate_threat(self, quantum_probability, raw_quantum_expval, snr_db=None):
        """
        Evaluates a detected sonar/radar target return.
        
        :param quantum_probability: Sigmoid output of the Quantum VQC [0.0, 1.0].
        :param raw_quantum_expval: Raw Pauli-Z expectation value [-1.0, 1.0].
        :param snr_db: Signal to noise ratio in dB (optional).
        :return: Dictionary containing threat analysis.
        """
        # Quantum Confidence: Distance from decision boundary (0.0)
        confidence_pct = float(np.clip(abs(raw_quantum_expval) * 100.0, 50.0, 99.9))
        
        # Threat Score (0 to 100 scale)
        threat_score = (self.w_prob * quantum_probability + self.w_conf * (confidence_pct / 100.0)) * 100.0
        
        # Target Classification & Threat Level assignment
        if quantum_probability >= 0.65:
            classification = "SUBMERGED METALLIC MINE"
            status = "HOSTILE / DETECTED"
            threat_level = "CRITICAL (RED)"
            action = "ALERT: INITIATE MINE COUNTERMEASURES / EVASIVE MANEUVER"
        elif quantum_probability >= 0.45:
            classification = "UNIDENTIFIED SUBSURFACE CONTACT"
            status = "SUSPICIOUS"
            threat_level = "ELEVATED (AMBER)"
            action = "CAUTION: INCREASE SENSOR DWELL TIME & FREQUENCY SWEEP"
        else:
            classification = "NATURAL SEAFLOOR ROCK / CLUTTER"
            status = "CLEAR / NON-THREAT"
            threat_level = "LOW (GREEN)"
            action = "NORMAL: MAINTAIN ACTIVE PATROL COURSE"

        return {
            "classification": classification,
            "status": status,
            "threat_level": threat_level,
            "threat_score": round(float(threat_score), 2),
            "confidence_pct": round(confidence_pct, 2),
            "quantum_prob": round(float(quantum_probability), 4),
            "action": action
        }
