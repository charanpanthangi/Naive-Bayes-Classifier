"""Data loading utilities for the Naive Bayes tutorial project."""
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Iris dataset as pandas structures.

    Returns:
        Tuple containing feature DataFrame and target Series.
    """
    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = iris.target
    return X, y


__all__ = ["load_iris_dataset"]
