"""Classical Radar and Sonar baseline processing and ML models."""

from .baselines import (
    ClassicalMatchedFilter,
    ClassicalSVM,
    ClassicalRandomForest,
    train_classical_baselines
)

__all__ = [
    "ClassicalMatchedFilter",
    "ClassicalSVM",
    "ClassicalRandomForest",
    "train_classical_baselines"
]
