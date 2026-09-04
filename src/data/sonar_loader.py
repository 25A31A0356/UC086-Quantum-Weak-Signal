"""
Sonar Dataset Preprocessor and Quantum Feature Formatter.
Preprocesses 60-band acoustic frequency modulation returns for Mines vs. Rocks classification.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from .kaggle_loader import KaggleDatasetManager


def load_sonar_dataset(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the Sonar dataset directly via Kaggle connection or provided path.
    
    Returns:
        X (np.ndarray): Shape (208, 60), 60 frequency band acoustic energy returns.
        y (np.ndarray): Shape (208,), binary labels (1 for Mine 'M', 0 for Rock 'R').
        df (pd.DataFrame): The raw dataframe.
    """
    if data_path is None or not str(data_path).strip():
        manager = KaggleDatasetManager()
        data_path = manager.fetch_sonar_dataset()
    
    # Load CSV (no header by default in UCI sonar dataset)
    df = pd.read_csv(data_path, header=None)
    
    # In some Kaggle versions, the last column might be labeled 'Class' or index 60
    if df.shape[1] == 61:
        X = df.iloc[:, :60].values.astype(np.float64)
        raw_labels = df.iloc[:, 60].values
    else:
        X = df.iloc[:, :-1].values.astype(np.float64)
        raw_labels = df.iloc[:, -1].values
    
    # Map 'M' (Mine) -> 1, 'R' (Rock) -> 0
    y = np.array([1 if str(label).strip().upper() == 'M' else 0 for label in raw_labels], dtype=np.int64)
    
    return X, y, df


def prepare_sonar_quantum_data(
    n_qubits: int = 10,
    test_size: float = 0.25,
    random_state: int = 42,
    scaling: str = "angle",
    data_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Prepare Sonar data for Quantum AI / ML models.
    Applies PCA dimensionality reduction to match QPU qubit register count,
    and scales features for quantum embedding circuits.

    Args:
        n_qubits (int): Number of quantum features / qubits (default 10).
        test_size (float): Proportion of dataset for test split (default 0.25).
        random_state (int): Seed for reproducibility.
        scaling (str): 
            - "angle": Scale to [0, pi] for Pauli rotation angle embeddings.
            - "amplitude": L2 normalize for amplitude state vector embedding.
            - "standard": Standard Gaussian normalization (mean 0, var 1).
        data_path (str, optional): Custom path to sonar CSV.

    Returns:
        X_train, X_test, y_train, y_test, pca_model
    """
    X, y, _ = load_sonar_dataset(data_path)

    # Train / test split first to avoid data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize before PCA
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    # Dimensionality reduction to n_qubits
    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Quantum Feature Scaling
    if scaling == "angle":
        # Scale to [0, pi] for RY / RZ quantum gates
        q_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = q_scaler.fit_transform(X_train_pca)
        X_test = q_scaler.transform(X_test_pca)
    elif scaling == "amplitude":
        # Normalize each sample to unit norm for amplitude embedding
        norms_train = np.linalg.norm(X_train_pca, axis=1, keepdims=True)
        norms_train[norms_train == 0] = 1.0
        X_train = X_train_pca / norms_train

        norms_test = np.linalg.norm(X_test_pca, axis=1, keepdims=True)
        norms_test[norms_test == 0] = 1.0
        X_test = X_test_pca / norms_test
    elif scaling == "standard":
        X_train, X_test = X_train_pca, X_test_pca
    else:
        raise ValueError(f"Unknown scaling method: {scaling}. Choose 'angle', 'amplitude', or 'standard'.")

    return X_train, X_test, y_train, y_test, pca
