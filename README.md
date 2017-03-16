[![Build Status](https://travis-ci.org/hammerlab/knnimpute.svg?branch=master)](https://travis-ci.org/hammerlab/knnimpute) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/knnimpute/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/knnimpute?branch=master)

# knnimpute
Implementations of kNN imputation in pure Python + NumPy, with a wrapper that
selects between them based on the number of neighbors, featurs, and missing
entries.

## Example

```python
from knnimpute import knn_impute
X_imputed = knn_impute(X, missing_mask=np.isnan(X), k=3)
```

## Algorithms
* Optimistic: assumes that most neighbors of a sample have whatever features it is missing
* Few Observed: assumes that many entries are missing
* Arg-Partition: relies on numpy.argpartition in the inner loop of imputation