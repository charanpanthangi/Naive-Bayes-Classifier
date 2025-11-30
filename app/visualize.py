"""Visualization utilities for Naive Bayes results."""
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


DEFAULT_OUT = Path("visualizations")


def plot_confusion_matrix(
    model_name: str,
    cm: np.ndarray,
    labels: Iterable[str],
    output_dir: Path = DEFAULT_OUT,
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        model_name: Name of the model variant.
        cm: Confusion matrix array.
        labels: Class labels for axes.
        output_dir: Directory where the plot will be stored.

    Returns:
        Path to the saved SVG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name.title()}NB")
    path = output_dir / f"confusion_matrix_{model_name}.svg"
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()
    return path


def plot_pca_predictions(
    model_name: str,
    model: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = DEFAULT_OUT,
) -> Path:
    """Project data to 2D with PCA and plot predicted classes.

    Args:
        model_name: Name of the model variant.
        model: Trained model.
        X_train: Training features for fitting PCA.
        X_test: Testing features for projection and prediction.
        y_train: Training labels.
        y_test: Testing labels.
        output_dir: Directory for saving the figure.

    Returns:
        Path to the saved SVG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap="viridis", alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of Predictions - {model_name.title()}NB")
    legend1 = plt.legend(*scatter.legend_elements(), title="Predicted")
    plt.gca().add_artist(legend1)
    path = output_dir / f"pca_predictions_{model_name}.svg"
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()
    return path


__all__ = ["plot_confusion_matrix", "plot_pca_predictions"]
