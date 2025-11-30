"""Model utilities for different Naive Bayes variants."""
from typing import Dict
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


def get_models() -> Dict[str, object]:
    """Create instances of the three Naive Bayes variants.

    Returns:
        Dictionary mapping a short name to a model instance.
    """
    return {
        "gaussian": GaussianNB(),
        "multinomial": MultinomialNB(),
        "bernoulli": BernoulliNB(),
    }


__all__ = ["get_models"]
