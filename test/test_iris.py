from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)
import numpy as np
from sklearn.datasets import load_iris

def _check_iris_imputation(_impute_fn):
    iris = load_iris()
    X = iris.data
    # some values missing only
    rng = np.random.RandomState(0)
    X_some_missing = X.copy()
    mask = np.abs(X[:, 2] - rng.normal(loc=5.5, scale=.7, size=X.shape[0])) < .6
    X_some_missing[mask, 3] = np.NaN
    X_imputed = _impute_fn(X_some_missing, np.isnan(X_some_missing), k=3)
    mean_abs_diff = np.mean(np.abs(X - X_imputed))
    print(mean_abs_diff)
    assert mean_abs_diff < 0.05, "Difference too big: %0.4f" % mean_abs_diff

def test_iris():
    for fn in (
            knn_impute_few_observed,
            knn_impute_with_argpartition,
            knn_impute_optimistic,
            knn_impute_reference):
        yield _check_iris_imputation, fn
