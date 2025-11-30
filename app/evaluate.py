"""Evaluation helpers for Naive Bayes classifiers."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MetricsDict = Dict[str, Dict[str, float | np.ndarray]]


def evaluate_models(
    models: Dict[str, object],
    features: Dict[str, tuple],
    y_train: pd.Series,
    y_test: pd.Series,
) -> MetricsDict:
    """Fit and evaluate each model using provided feature variants.

    Args:
        models: Dictionary of model instances.
        features: Dictionary mapping variant name to (X_train, X_test).
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        Nested dictionary of metrics per model.
    """
    results: MetricsDict = {}
    for name, model in models.items():
        X_train, X_test = features[name]
        fitted_model = model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    return results


__all__ = ["evaluate_models", "MetricsDict"]
