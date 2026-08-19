# Principal Component Analysis (PCA)

`PCA` is used to reduce the dimensionality of local descriptors of VLAD and Fisher Vector before clustering in
order to reduce the size of the final embedding.

## Parameters (`pca_params`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_components` | (required) | Number of components to keep. Must be at most `min(n_samples, n_features)` of the training descriptors. |
| `whiten` | `False` | Scale each projected component to unit variance. Components with near-zero variance (rank-deficient descriptors) are floored at machine epsilon so the output stays finite. |
| `svd_solver` | `"auto"` | `"full"` (economy SVD), `"covariance_eigh"` (eigendecomposition of the feature covariance, fastest for many samples with few features), `"arpack"` (truncated SVD, computes only `n_components` singular triplets), or `"auto"`, which picks between them based on the training shape. |
| `tol` | `0.0` | Convergence tolerance of the `"arpack"` solver (0 means machine precision). Ignored by the other solvers. |
| `rng` | `None` | Seed (`int`) or `numpy.random.Generator` for the `"arpack"` solver's starting vector. Ignored by the other solvers. |
