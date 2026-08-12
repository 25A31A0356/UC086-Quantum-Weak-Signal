import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PARAMETERS
# ============================================================

INPUT_FILE = "features/radar_features.csv"

ENGINEERED_OUTPUT = "features/engineered_features.csv"
TRAIN_INDEX_OUTPUT = "features/train_indices.npy"
TEST_INDEX_OUTPUT = "features/test_indices.npy"
FEATURE_LIST_OUTPUT = "features/selected_features.json"
SCALER_OUTPUT = "features/feature_scaler.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# SELECTED FEATURES
# ============================================================

SELECTED_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "rms",
    "peak_to_peak",
    "energy",
    "dominant_frequency",
    "dominant_magnitude",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_features(file_path):
    """Load extracted radar features."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_features(df):
    """Check the extracted feature dataset."""

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if df.isnull().sum().sum() != 0:
        raise ValueError(
            "Feature dataset contains missing values."
        )

    numeric_data = df.select_dtypes(
        include=np.number
    )

    if np.isinf(numeric_data).sum().sum() != 0:
        raise ValueError(
            "Feature dataset contains infinite values."
        )

    required_columns = (
        SELECTED_FEATURES + ["label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Select the features required by ML models."""

    engineered_df = df[
        SELECTED_FEATURES + ["label"]
    ].copy()

    return engineered_df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_split(df):
    """Create reproducible stratified train/test indices."""

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    return train_indices, test_indices


# ============================================================
# SCALE FEATURES
# ============================================================

def create_scaler(df, train_indices):
    """
    Fit scaler using training data only.
    """

    scaler = StandardScaler()

    scaler.fit(
        df.iloc[
            train_indices
        ][SELECTED_FEATURES]
    )

    return scaler


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    engineered_df,
    train_indices,
    test_indices,
    scaler
):

    os.makedirs(
        "features",
        exist_ok=True
    )

    engineered_df.to_csv(
        ENGINEERED_OUTPUT,
        index=False
    )

    np.save(
        TRAIN_INDEX_OUTPUT,
        train_indices
    )

    np.save(
        TEST_INDEX_OUTPUT,
        test_indices
    )

    with open(
        FEATURE_LIST_OUTPUT,
        "w"
    ) as file:

        json.dump(
            SELECTED_FEATURES,
            file,
            indent=4
        )

    joblib.dump(
        scaler,
        SCALER_OUTPUT
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("===================================")
    print("M3 FEATURE ENGINEERING")
    print("===================================")

    # Load
    df = load_features(
        INPUT_FILE
    )

    print(
        "Input shape:",
        df.shape
    )

    # Validate
    validate_features(df)

    print(
        "Validation: PASSED"
    )

    # Engineer
    engineered_df = engineer_features(
        df
    )

    print(
        "Engineered shape:",
        engineered_df.shape
    )

    # Split
    train_indices, test_indices = (
        create_split(engineered_df)
    )

    print(
        "Training samples:",
        len(train_indices)
    )

    print(
        "Testing samples:",
        len(test_indices)
    )

    # Scale
    scaler = create_scaler(
        engineered_df,
        train_indices
    )

    print(
        "Scaler fitted using training data only."
    )

    # Save
    save_outputs(
        engineered_df,
        train_indices,
        test_indices,
        scaler
    )

    print(
        "\nM3 feature engineering completed."
    )


if __name__ == "__main__":
    main()
