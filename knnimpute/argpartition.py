# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division
import time

import numpy as np
from six.moves import range

from .common import knn_initialize

def knn_impute_with_argpartition(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.

    This version is a simpler algorithm meant primarily for testing but
    surprisingly it's faster for many (but not all) dataset sizes, particularly
    when most of the columns are missing in any given row. The crucial
    bottleneck is the call to numpy.argpartition for every missing element
    in the array.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool

    Returns a row-major copy of X with imputed values.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)
    # D[~np.isfinite(D)] = very_large_value
    D_reciprocal = 1.0 / D
    neighbor_weights = np.zeros(k, dtype="float32")
    dot = np.dot
    for i in range(n_rows):
        missing_indices = np.where(missing_mask[i])[0]

        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        d = D[i, :]
        inv_d = D_reciprocal[i, :]
        for j in missing_indices:
            column = X[:, j]
            rows_missing_feature = missing_mask_column_major[:, j]
            d_copy = d.copy()
            # d_copy[rows_missing_feature] = very_large_value
            d_copy[rows_missing_feature] = np.inf
            neighbor_indices = np.argpartition(d_copy, k)[:k]
            if len(neighbor_indices) > 0:
                neighbor_weights = inv_d[neighbor_indices]
                X_row_major[i, j] = (
                    dot(column[neighbor_indices], neighbor_weights) /
                    neighbor_weights.sum()
                )
    return X_row_major
