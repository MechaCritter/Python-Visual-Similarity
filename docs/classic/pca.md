# Principal Component Analysis (PCA)

Both classic embedders multiply their output size by the local descriptor dimension
`D`, so a 128-dimensional RootSIFT descriptor with 256 clusters already gives you a
32768-dimensional VLAD vector. Running PCA over the descriptors before clustering was
the standard way to keep that under control: halve `D` and you halve the embedding.
It usually helps accuracy too, since it decorrelates the descriptor dimensions that
the diagonal-covariance GMM behind the Fisher Vector assumes are independent anyway.

Pass `pca_params` to either embedder to switch it on. The PCA is fitted first, inside
`learn`, and the clustering model then learns from the projected descriptors:

```python
vlad = VLADEmbedder(n_clusters=256, pca_params={"n_components": 64})
vlad.learn(images)
```

## Parameters (`pca_params`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_components` | (required) | Number of components to keep. Must be at most `min(n_samples, n_features)` of the training descriptors. |
| `whiten` | `False` | Scale each projected component to unit variance. Components with near-zero variance (rank-deficient descriptors) are floored at machine epsilon so the output stays finite. |
| `svd_solver` | `"auto"` | `"full"` (economy SVD), `"covariance_eigh"` (eigendecomposition of the feature covariance, fastest for many samples with few features), `"arpack"` (truncated SVD, computes only `n_components` singular triplets), or `"auto"`, which picks between them based on the training shape. |
| `tol` | `0.0` | Convergence tolerance of the `"arpack"` solver (0 means machine precision). Ignored by the other solvers. |
| `rng` | `None` | Seed (`int`) or `numpy.random.Generator` for the `"arpack"` solver's starting vector. Ignored by the other solvers. |
