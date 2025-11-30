"""End-to-end pipeline to train, evaluate, and visualize Naive Bayes models."""
from typing import Dict

from app.data import load_iris_dataset
from app.evaluate import evaluate_models
from app.model import get_models
from app.preprocess import transform_features_for_variants, split_data
from app.visualize import plot_confusion_matrix, plot_pca_predictions


def run_pipeline(scale_gaussian: bool = False) -> Dict[str, Dict[str, float]]:
    """Execute the full training and evaluation pipeline.

    Args:
        scale_gaussian: Whether to standardize features for GaussianNB.

    Returns:
        Metrics dictionary per model.
    """
    X, y = load_iris_dataset()
    split = split_data(X, y)
    variant_features = transform_features_for_variants(split, scale_gaussian=scale_gaussian)

    models = get_models()
    features_dict = {
        "gaussian": variant_features.gaussian,
        "multinomial": variant_features.multinomial,
        "bernoulli": variant_features.bernoulli,
    }
    metrics = evaluate_models(models, features_dict, split.y_train, split.y_test)

    labels = ["setosa", "versicolor", "virginica"]

    for name, result in metrics.items():
        cm_path = plot_confusion_matrix(name, result["confusion_matrix"], labels)
        pca_path = plot_pca_predictions(
            name,
            models[name],
            features_dict[name][0],
            features_dict[name][1],
            split.y_train,
            split.y_test,
        )
        print(f"Saved {name} confusion matrix to {cm_path}")
        print(f"Saved {name} PCA plot to {pca_path}")

    print("\nModel comparison (higher is better):")
    for name, scores in metrics.items():
        print(f"\n{name.title()}NB")
        print(f"Accuracy : {scores['accuracy']:.3f}")
        print(f"Precision: {scores['precision']:.3f}")
        print(f"Recall   : {scores['recall']:.3f}")
        print(f"F1-score : {scores['f1']:.3f}")

    return metrics


if __name__ == "__main__":
    run_pipeline(scale_gaussian=False)
