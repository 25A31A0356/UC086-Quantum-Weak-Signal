"""Evaluation metrics, benchmarks, and radar/sonar visualization utilities."""

from .metrics import (
    evaluate_detection_metrics,
    compute_clutter_rejection_ratio,
    run_snr_sweep_benchmark,
    plot_roc_comparison,
    plot_snr_sweep_results,
    plot_radar_spectrogram
)

__all__ = [
    "evaluate_detection_metrics",
    "compute_clutter_rejection_ratio",
    "run_snr_sweep_benchmark",
    "plot_roc_comparison",
    "plot_snr_sweep_results",
    "plot_radar_spectrogram"
]
