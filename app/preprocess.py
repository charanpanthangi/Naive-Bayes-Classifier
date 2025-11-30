"""Preprocessing utilities including train/test split and feature transforms."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler


@dataclass
class SplitData:
    """Container for train/test splits."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class VariantFeatures:
    """Feature representations for each Naive Bayes variant."""
    gaussian: Tuple[np.ndarray, np.ndarray]
    multinomial: Tuple[np.ndarray, np.ndarray]
    bernoulli: Tuple[np.ndarray, np.ndarray]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Split features and labels into train and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion for the test split.
        random_state: Seed for reproducibility.

    Returns:
        SplitData with training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def transform_features_for_variants(
    split: SplitData, scale_gaussian: bool = False
) -> VariantFeatures:
    """Prepare feature matrices tailored for each Naive Bayes variant.

    GaussianNB can work with raw or standardized features, while MultinomialNB
    and BernoulliNB expect non-negative inputs. We therefore apply a Min-Max
    scaler to keep values between 0 and 1, and a binarizer for BernoulliNB.

    Args:
        split: SplitData containing train/test splits.
        scale_gaussian: Whether to standardize features for GaussianNB.

    Returns:
        VariantFeatures with (X_train, X_test) tuples per variant.
    """
    X_train_gaussian = split.X_train.copy()
    X_test_gaussian = split.X_test.copy()

    if scale_gaussian:
        scaler = StandardScaler()
        X_train_gaussian = scaler.fit_transform(X_train_gaussian)
        X_test_gaussian = scaler.transform(X_test_gaussian)

    minmax_scaler = MinMaxScaler()
    X_train_mnm = minmax_scaler.fit_transform(split.X_train)
    X_test_mnm = minmax_scaler.transform(split.X_test)

    binarizer = Binarizer(threshold=0.5)
    X_train_bin = binarizer.fit_transform(X_train_mnm)
    X_test_bin = binarizer.transform(X_test_mnm)

    return VariantFeatures(
        gaussian=(
            np.asarray(X_train_gaussian),
            np.asarray(X_test_gaussian),
        ),
        multinomial=(X_train_mnm, X_test_mnm),
        bernoulli=(X_train_bin, X_test_bin),
    )


__all__ = ["SplitData", "VariantFeatures", "split_data", "transform_features_for_variants"]
