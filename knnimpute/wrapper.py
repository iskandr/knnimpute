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

import numpy as np
from .optimistic import knn_impute_optimistic
from .argpartition import knn_impute_with_argpartition
from .few_observed_entries import knn_impute_few_observed

def knn_impute(X, missing_mask=None, k=1):
    """
    Perform kNN imputation using one of three underlying algorithms:
        1) "Arg-Partition": relies on the numpy.argpartition function in
            inner loop of imputation.
        2) "Few Observed": fastest when many entries are misssing
        3) "Optimistic": assumes that all neighbors have a missing feature
            and then follows a slower path if that assumption is false.

    The decision boundary between these was determined via a very small
    sample size logistic regression, can probably be improved! Features for
    the logistic regression are:
        - number of neighbors
        - number of feature columns
        - fraction missing
    """

    if missing_mask is None:
        missing_mask = np.isnan(X)

    fraction_missing = missing_mask.mean()
    problem_features = np.array([k, X.shape[1], fraction_missing])
    decision_coefficients = np.array(
        [[0.035, -0.023, -0.383],
         [0.114, 0.008, 0.486],
         [-0.149, 0.0148, -0.103]])
    decision_intercepts = np.array([1.498, -0.327, -1.171])
    scores = np.dot(problem_features, decision_coefficients.T) + decision_intercepts
    functions = [
        knn_impute_with_argpartition,
        knn_impute_few_observed,
        knn_impute_optimistic
    ]
    choice_idx = np.argmax(scores)
    best_impute_fn = functions[choice_idx]
    return best_impute_fn(X=X, missing_mask=missing_mask, k=k)
