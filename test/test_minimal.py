
import numpy as np

from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

def _run_knn_with_two_rows(impute_fn):
    X = np.array([[1, 1, np.NaN], [1, 1, 1]])
    result = impute_fn(X, np.isnan(X), k=1)
    assert not np.isnan(result).any(), \
        "Basic example did not get imputed: %s" % result

def test_knn_minimal():
    for fn in (
            knn_impute_few_observed,
            knn_impute_with_argpartition,
            knn_impute_optimistic,
            knn_impute_reference):
        yield _run_knn_with_two_rows, fn
