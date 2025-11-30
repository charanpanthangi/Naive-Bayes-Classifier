from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.data import load_iris_dataset
from app.evaluate import evaluate_models
from app.model import get_models
from app.preprocess import split_data, transform_features_for_variants


def test_evaluate_returns_metrics():
    X, y = load_iris_dataset()
    split = split_data(X, y, test_size=0.25, random_state=1)
    features = transform_features_for_variants(split)

    models = get_models()
    features_dict = {
        "gaussian": features.gaussian,
        "multinomial": features.multinomial,
        "bernoulli": features.bernoulli,
    }

    results = evaluate_models(models, features_dict, split.y_train, split.y_test)
    for metrics in results.values():
        for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
            assert key in metrics
