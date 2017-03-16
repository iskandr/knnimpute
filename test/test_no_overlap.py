import numpy as np
from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

def _attempt_knn_without_overlapping_entries(impute_fn):
    X_no_overlap = np.array([
        [1.0, np.NaN],
        [np.NaN, 2.0]])
    result = impute_fn(X_no_overlap, np.isnan(X_no_overlap), k=1)
    print(result)

    assert np.allclose(result[0, 0], 1.0)
    # since this is a degenerate case we allow either leaving it as NaN
    # or filling with the column mean. Maybe it's worth standardizing on
    # pre-filling with the column mean in knn_initialize
    assert np.isnan(result[0, 1]) or np.allclose(result[0, 1], 2.0)
    assert np.isnan(result[1, 0]) or np.allclose(result[1, 0], 1.0)
    assert np.allclose(result[1, 1], 2.0)

def test_imputation_with_no_overlapping_samples():
    for fn in (
            knn_impute_few_observed,
            knn_impute_with_argpartition,
            knn_impute_optimistic,
            knn_impute_reference):
        yield _attempt_knn_without_overlapping_entries, fn
