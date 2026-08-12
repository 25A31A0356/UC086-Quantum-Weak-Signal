import numpy as np
import pandas as pd
import joblib


# ============================================================
# LOAD M3 DATA
# ============================================================

df = pd.read_csv(
    "../features/engineered_features.csv"
)

train_indices = np.load(
    "../features/train_indices.npy"
)

test_indices = np.load(
    "../features/test_indices.npy"
)

scaler = joblib.load(
    "../features/feature_scaler.pkl"
)


# ============================================================
# SELECT 4 FEATURES FOR QUANTUM CIRCUIT
# ============================================================

QUANTUM_FEATURES = [
    "mean",
    "std",
    "rms",
    "energy"
]


X = df[QUANTUM_FEATURES]

y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train = X.iloc[train_indices]

X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]

y_test = y.iloc[test_indices]


# ============================================================
# IMPORTANT:
# The M3 scaler was trained on all 9 features.
#
# Therefore we cannot directly use it on only 4 columns.
# We create a scaler specifically for the selected
# quantum features using TRAINING DATA ONLY.
# ============================================================

from sklearn.preprocessing import StandardScaler

quantum_scaler = StandardScaler()

X_train_scaled = quantum_scaler.fit_transform(
    X_train
)

X_test_scaled = quantum_scaler.transform(
    X_test
)


# ============================================================
# CONVERT FEATURES TO QUANTUM-FRIENDLY RANGE
# ============================================================

X_train_quantum = np.pi * np.tanh(
    X_train_scaled
)

X_test_quantum = np.pi * np.tanh(
    X_test_scaled
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("QUANTUM DATA PREPARATION")
print("=" * 60)

print("Selected features:")
for feature in QUANTUM_FEATURES:
    print("-", feature)

print("\nNumber of qubits:", len(QUANTUM_FEATURES))

print(
    "\nTraining shape:",
    X_train_quantum.shape
)

print(
    "Testing shape:",
    X_test_quantum.shape
)

print(
    "\nQuantum value range:",
    X_train_quantum.min(),
    "to",
    X_train_quantum.max()
)


# ============================================================
# SAVE PREPARED DATA
# ============================================================

np.save(
    "X_train_quantum.npy",
    X_train_quantum
)

np.save(
    "X_test_quantum.npy",
    X_test_quantum
)

np.save(
    "y_train_quantum.npy",
    y_train.to_numpy()
)

np.save(
    "y_test_quantum.npy",
    y_test.to_numpy()
)

joblib.dump(
    quantum_scaler,
    "quantum_scaler.pkl"
)


print("\nSaved files:")
print("X_train_quantum.npy")
print("X_test_quantum.npy")
print("y_train_quantum.npy")
print("y_test_quantum.npy")
print("quantum_scaler.pkl")
