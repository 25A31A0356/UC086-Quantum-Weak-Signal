import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


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
# FEATURES
# ============================================================

FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "rms",
    "peak_to_peak",
    "energy",
    "dominant_frequency",
    "dominant_magnitude"
]

X = df[FEATURES]
y = df["label"]


# ============================================================
# TRAIN / TEST DATA
# ============================================================

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# ============================================================
# SCALE USING M3 SCALER
# ============================================================

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            random_state=42,
            max_iter=1000
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    results.append({

        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1":
            f1_score(
                y_test,
                predictions
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("CLASSICAL ML RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results.csv",
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["F1"].idxmax(),
    "Model"
]

best_model = models[
    best_model_name
]


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "best_classical_model.pkl"
)


print("\n" + "=" * 60)
print("M4 COMPLETE")
print("=" * 60)

print(
    "Best model:",
    best_model_name
)

print(
    "Saved: results.csv"
)

print(
    "Saved: best_classical_model.pkl"
)
