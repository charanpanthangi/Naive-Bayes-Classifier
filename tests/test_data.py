from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.data import load_iris_dataset


def test_load_iris_dataset_shapes():
    X, y = load_iris_dataset()
    assert len(X) == len(y)
    assert X.shape[1] == 4  # sepal length/width, petal length/width
    assert y.nunique() == 3
