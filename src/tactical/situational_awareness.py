"""
Tactical Situational Awareness Dashboard for Quantum Radar & Sonar.
Renders real-time polar radar scopes, threat metrics, and alert panels.
"""

import matplotlib.pyplot as plt
import numpy as np


class TacticalDashboard:
    def __init__(self):
        plt.style.use('dark_background')

    def render_tactical_hud(self, target_reports, current_time="T+00:14:32"):
        """
        Renders a 4-panel Tactical Maritime Defense Situational Awareness Dashboard.
        :param target_reports: List of threat detection report dictionaries.
        :param current_time: Current simulation timestamp.
        """
        fig = plt.figure(figsize=(15, 8), dpi=120)
        fig.patch.set_facecolor('#0a0f1d')

        # Panel 1: Polar Radar / Sonar PPI Scope
        ax_polar = plt.subplot2grid((2, 3), (0, 0), rowspan=2, projection='polar', facecolor='#06101e')
        theta = np.linspace(0, 2 * np.pi, 200)
        # Range rings
        for r in [2, 4, 6, 8, 10]:
            ax_polar.plot(theta, [r]*200, color='#00ffcc', alpha=0.2, lw=0.8, linestyle='--')
        
        # Sweep beam line
        sweep_angle = np.pi / 4
        ax_polar.plot([sweep_angle, sweep_angle], [0, 10], color='#00ffcc', lw=1.5, alpha=0.8)
        
        # Plot contacts on scope
        colors = {"CRITICAL (RED)": "#ff3333", "ELEVATED (AMBER)": "#ffaa00", "LOW (GREEN)": "#00ff66"}
        for i, rep in enumerate(target_reports[:5]):
            angle = (i * 1.3) % (2 * np.pi)
            distance = 3.0 + (i * 1.5)
            c = colors.get(rep["threat_level"], "#00ffcc")
            ax_polar.scatter([angle], [distance], color=c, s=120, edgecolors='white', lw=1.5, zorder=5)
            ax_polar.text(angle + 0.1, distance, f"TGT-{i+1}\n[{rep['threat_score']}]", color=c, fontsize=8, fontweight='bold')

        ax_polar.set_title("🌐 QUANTUM ACTIVE PPI SCOPE (10 km)", color='#00ffcc', fontsize=12, fontweight='bold', pad=15)
        ax_polar.tick_params(colors='#00ffcc', labelsize=8)
        ax_polar.grid(color='#00ffcc', alpha=0.15)

        # Panel 2: Threat Scores Comparison Bar Chart
        ax_scores = plt.subplot2grid((2, 3), (0, 1), facecolor='#06101e')
        tgt_names = [f"TGT-{i+1}" for i in range(len(target_reports))]
        scores = [r["threat_score"] for r in target_reports]
        bar_colors = [colors.get(r["threat_level"], "#00ffcc") for r in target_reports]
        
        bars = ax_scores.bar(tgt_names, scores, color=bar_colors, alpha=0.85, edgecolor='white', lw=0.8)
        ax_scores.set_ylim(0, 100)
        ax_scores.set_ylabel("Threat Score (0-100)", color='white', fontsize=9, fontweight='bold')
        ax_scores.set_title("⚠️ QUANTUM THREAT ASSESSMENT", color='white', fontsize=11, fontweight='bold')
        ax_scores.tick_params(colors='white', labelsize=8)
        ax_scores.grid(axis='y', linestyle=':', alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax_scores.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}", ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

        # Panel 3: Quantum Confidence Gauge
        ax_conf = plt.subplot2grid((2, 3), (0, 2), facecolor='#06101e')
        confidences = [r["confidence_pct"] for r in target_reports]
        ax_conf.plot(tgt_names, confidences, marker='o', color='#00ccff', lw=2.0, markersize=8)
        ax_conf.set_ylim(40, 100)
        ax_conf.set_ylabel("Quantum Confidence (%)", color='white', fontsize=9, fontweight='bold')
        ax_conf.set_title("⚛️ QUANTUM STATE FIDELITY", color='white', fontsize=11, fontweight='bold')
        ax_conf.tick_params(colors='white', labelsize=8)
        ax_conf.grid(True, linestyle=':', alpha=0.3)

        # Panel 4: Tactical Command & Action Feed
        ax_feed = plt.subplot2grid((2, 3), (1, 1), colspan=2, facecolor='#06101e')
        ax_feed.axis('off')
        
        feed_text = f"TACTICAL SITUATIONAL AWARENESS FEED | SIM TIME: {current_time}\n" + "─" * 68 + "\n"
        for i, rep in enumerate(target_reports[:4]):
            status_symbol = "🔴" if "RED" in rep["threat_level"] else ("🟡" if "AMBER" in rep["threat_level"] else "🟢")
            feed_text += f"{status_symbol} [TGT-{i+1}] {rep['classification']} | Conf: {rep['confidence_pct']}% | Score: {rep['threat_score']}/100\n"
            feed_text += f"   └── ACTION: {rep['action']}\n\n"

        ax_feed.text(0.02, 0.95, feed_text, color='#00ffcc', fontfamily='monospace', fontsize=8.5, va='top')

        plt.suptitle("🛡️ DEFENSE SITUATIONAL AWARENESS SYSTEM — QUANTUM RADAR & SONAR", color='white', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
