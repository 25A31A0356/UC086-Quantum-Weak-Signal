import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD M3 ENGINEERED DATA
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


print("=" * 60)
print("M5 QUANTUM DATA PREPARATION")
print("=" * 60)

print("Engineered features loaded")
print("Train indices loaded")
print("Test indices loaded")


# ============================================================
# 2. SELECT FEATURES FOR QUANTUM CIRCUIT
# ============================================================

QUANTUM_FEATURES = [
    "mean",
    "std",
    "rms",
    "energy"
]

X = df[QUANTUM_FEATURES]
y = df["label"]


print("\nSelected quantum features:")

for feature in QUANTUM_FEATURES:
    print("-", feature)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. QUANTUM-SPECIFIC SCALING
# ============================================================

quantum_scaler = StandardScaler()

X_train_scaled = quantum_scaler.fit_transform(
    X_train
)

X_test_scaled = quantum_scaler.transform(
    X_test
)


# ============================================================
# 5. CONVERT FEATURES TO QUANTUM ANGLES
# ============================================================

X_train_quantum = (
    np.pi * np.tanh(X_train_scaled)
)

X_test_quantum = (
    np.pi * np.tanh(X_test_scaled)
)


# ============================================================
# 6. VERIFY DATA
# ============================================================

print("\n" + "=" * 60)
print("QUANTUM DATA")
print("=" * 60)

print(
    "Number of quantum features:",
    len(QUANTUM_FEATURES)
)

print(
    "Training shape:",
    X_train_quantum.shape
)

print(
    "Testing shape:",
    X_test_quantum.shape
)

print(
    "Quantum value range:",
    X_train_quantum.min(),
    "to",
    X_train_quantum.max()
)


# ============================================================
# 7. SAVE QUANTUM DATA
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


# ============================================================
# 8. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("M5 QUANTUM DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nGenerated:")

print("X_train_quantum.npy")
print("X_test_quantum.npy")
print("y_train_quantum.npy")
print("y_test_quantum.npy")
print("quantum_scaler.pkl")
