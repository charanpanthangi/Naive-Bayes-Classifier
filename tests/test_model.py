from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.data import load_iris_dataset
from app.model import get_models
from app.preprocess import split_data, transform_features_for_variants


def test_models_fit_and_predict():
    X, y = load_iris_dataset()
    split = split_data(X, y, test_size=0.3, random_state=0)
    features = transform_features_for_variants(split, scale_gaussian=False)

    models = get_models()
    features_dict = {
        "gaussian": features.gaussian,
        "multinomial": features.multinomial,
        "bernoulli": features.bernoulli,
    }

    for name, model in models.items():
        X_train, X_test = features_dict[name]
        fitted = model.fit(X_train, split.y_train)
        preds = fitted.predict(X_test)
        assert len(preds) == len(split.y_test)
