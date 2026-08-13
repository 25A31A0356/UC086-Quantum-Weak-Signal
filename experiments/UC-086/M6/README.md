# UC-086 — M6 Validation and Reproducibility

## 1. Objective

M6 validates and consolidates the classical and quantum machine
learning results obtained during the earlier stages of the
UC-086 prototype.

The main objectives are:

- verify the available M4 classical results
- verify the M5 quantum results
- create a unified classical-vs-quantum comparison
- calculate metric differences
- investigate the unusually high classical performance
- analyze performance across signal-to-noise ratio (SNR)
- organize reproducible M6 outputs
- document the experimental limitations

M6 does not retrain or modify the original M4/M5 results.

---

## 2. Dataset

The engineered feature dataset contains:

- 1000 samples
- 9 classification features
- 500 Class 0 samples
- 500 Class 1 samples

The engineered features are:

1. mean
2. std
3. max
4. min
5. rms
6. peak_to_peak
7. energy
8. dominant_frequency
9. dominant_magnitude

Additional project data:

- radar_features.csv: 1000 × 14
- radar_metadata.csv: 1000 × 3

---

## 3. SNR Distribution

The dataset contains five SNR levels:

- 10 dB
- 5 dB
- 0 dB
- -5 dB
- -10 dB

Each SNR level contains:

- 100 Class 0 samples
- 100 Class 1 samples

Therefore:

- 200 samples per SNR
- 1000 samples total
- balanced classes at every SNR level

---

## 4. Classical Results

The saved classical results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| KNN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The best classical model by F1 is:

**Logistic Regression**

---

## 5. Quantum Result

The evaluated quantum model is:

**Variational Quantum Classifier**

Results:

| Metric | Score |
|---|---:|
| Accuracy | 0.475 |
| Precision | 0.476636 |
| Recall | 0.510 |
| F1 | 0.492754 |
| ROC-AUC | 0.4783 |

---

## 6. Classical vs Quantum Comparison

The best classical model is Logistic Regression.

Classical F1:

1.000000

Quantum F1:

0.492754

Difference:

-0.507246

The quantum F1 is approximately:

49.28%

of the best classical F1.

The current experimental configuration therefore shows
substantially higher performance for the evaluated classical
models than for the evaluated Variational Quantum Classifier.

This conclusion applies specifically to the current UC-086
prototype dataset and experimental configuration.

It should not be interpreted as a general statement that
classical machine learning always outperforms quantum machine
learning.

---

## 7. Investigation of Perfect Classical Performance

Because all four classical models achieved 1.0 across the
reported metrics, M6 investigated feature separation.

A single-feature five-fold cross-validation test produced:

| Feature | Mean CV Accuracy |
|---|---:|
| dominant_magnitude | 1.000 |
| dominant_frequency | 0.999 |
| energy | 0.575 |
| rms | 0.554 |
| std | 0.552 |
| mean | 0.493 |
| min | 0.402 |
| max | 0.365 |
| peak_to_peak | 0.360 |

This shows that two frequency-domain features are extremely
strong predictors of the class.

---

## 8. Feature-Label Correlation

The strongest feature-label correlations were:

| Feature | Correlation |
|---|---:|
| dominant_magnitude | 0.976020 |
| dominant_frequency | -0.696665 |
| std | 0.223491 |
| rms | 0.223295 |
| max | 0.147544 |
| peak_to_peak | 0.142580 |
| min | -0.135082 |
| energy | 0.133286 |
| mean | -0.029906 |

The strongest association is observed for
dominant_magnitude.

---

## 9. Critical Feature Investigation

### dominant_frequency

Class 0:

- Mean: 249.5
- Standard deviation: 145.411
- Minimum: 1.0
- Maximum: 500.0
- Median: 237.5

Class 1:

- Mean: 50.0
- Standard deviation: 0.0
- Minimum: 50.0
- Maximum: 50.0
- Median: 50.0

### dominant_magnitude

Class 0:

- Mean: 79.1305
- Standard deviation: 60.5446
- Minimum: 15.0847
- Maximum: 240.4266
- Median: 57.7368

Class 1:

- Mean: 501.2818
- Standard deviation: 27.8490
- Minimum: 381.4097
- Maximum: 644.5020
- Median: 500.4093

These distributions explain why the classical models can
separate the two classes so effectively.

---

## 10. Signal Generation Investigation

The signal generator defines:

SAMPLE_RATE = 1000

DURATION = 1.0

SIGNAL_FREQUENCY = 50

The clean signal is generated as a 50 Hz sinusoidal signal.

The generator also contains:

- controlled signal-plus-noise generation
- noise-only sample generation
- SNR-controlled noise
- SNR levels of 10, 5, 0, -5 and -10 dB

The investigation showed:

Class 1 samples contain a fixed 50 Hz signal component.

Class 0 samples are generated from the noise-only process.

This explains the extremely strong separation in
dominant_frequency and dominant_magnitude.

The investigation does not establish conventional train/test
data leakage from the evidence examined in M6.

Instead, it demonstrates that the current synthetic signal
generation creates a highly separable classification problem.

---

## 11. SNR Robustness

Logistic Regression was evaluated separately at each SNR level
using the saved train/test split.

| SNR | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| -10 dB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| -5 dB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0 dB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 dB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 dB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The classical model remained perfectly separable across all
tested SNR levels in the current prototype dataset.

Note that the number of test samples per SNR is lower than the
full 200 samples per SNR because the existing fixed train/test
split was used.

---

## 12. Main M6 Findings

### Finding 1 — Classical models perform extremely well

All four evaluated classical models achieved perfect reported
metrics on the current dataset.

### Finding 2 — Quantum performance is lower

The evaluated Variational Quantum Classifier achieved:

Accuracy = 0.475

F1 = 0.492754

ROC-AUC = 0.4783

### Finding 3 — Feature separation explains classical performance

dominant_magnitude and dominant_frequency almost independently
separate the two classes.

### Finding 4 — SNR does not explain the perfect result

The dataset is balanced at every SNR level, and Logistic
Regression achieved perfect performance at every tested SNR.

### Finding 5 — The synthetic problem is highly separable

The signal-generation setup uses a fixed 50 Hz signal component
for signal-present samples, making the current prototype easier
to classify.

---

## 13. Limitations

The current experiment has several important limitations.

1. The dataset is synthetic.
2. The signal-present class contains a fixed 50 Hz component.
3. Frequency-domain features provide extremely strong class
   separation.
4. The classical models therefore solve the current prototype
   classification problem very easily.
5. The quantum model was evaluated under the existing M5
   configuration and did not outperform the classical baseline.
6. The experiment does not establish quantum advantage.
7. The experiment does not establish that classical methods
   will always outperform quantum methods.
8. A harder and more realistic dataset would be required for
   stronger general conclusions.

---

## 14. Reproducibility Files

Important files used or generated during M6 include:

### Core data

- engineered_features.csv
- radar_features.csv
- radar_metadata.csv
- signal_generator.py

### Existing M4/M5 results

- results.csv
- quantum_results.csv
- classical_vs_quantum.csv

### Preprocessing and split

- train_indices.npy
- test_indices.npy
- feature_scaler.pkl

### M6 results

- final_classical_vs_quantum.csv
- m6_performance_comparison.csv
- m6_classical_performance_by_snr.csv
- UC086_M6_final_results_summary.csv
- UC086_M6_SNR_robustness_summary.csv
- UC086_M6_consolidated_results.csv

### M6 plots

- m6_classical_performance_vs_snr.png
- UC086_M6_final_classical_vs_quantum_plot.png
- UC086_M6_final_snr_robustness_plot.png

### M6 documentation

- UC086_M6_experiment_summary.txt

---

## 15. M6 Status

M6 validation and reproducibility work is complete.

Completed activities:

- file verification
- dataset verification
- classical result verification
- quantum result verification
- classical-vs-quantum comparison
- metric difference calculation
- performance visualization
- feature separation analysis
- feature-label correlation analysis
- signal generation inspection
- SNR distribution analysis
- SNR robustness analysis
- consolidated result creation
- final M6 documentation

---

## 16. Final M6 Conclusion

The UC-086 prototype successfully produced a reproducible
classical-versus-quantum comparison.

The evaluated classical models substantially outperformed the
evaluated Variational Quantum Classifier on the current
synthetic dataset.

The investigation showed that this result is strongly influenced
by the highly separable engineered features, particularly
dominant_frequency and dominant_magnitude.

Therefore, the M6 result should be presented as a validation and
baseline comparison of the current prototype rather than as
evidence of a universal classical or quantum advantage.

A future experiment should use a more challenging and realistic
signal-generation setup or real radar data before making broader
claims about quantum machine-learning performance.

============================================================
END OF UC-086 M6
============================================================
