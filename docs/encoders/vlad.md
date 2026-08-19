# VLADEmbedder

File: [`vlad.py`](../../pyvisim/encoders/vlad.py)

VLAD (Vector of Locally Aggregated Descriptors) embeds an image into a vector of
shape `(K * D,)`, where `K` is the number of KMeans clusters and `D` is the local
descriptor dimension (after optional PCA).

## Constructing one

VLAD always clusters with K-Means, so you configure that model through the embedder:

```python
from pyvisim.encoders import VLADEmbedder

vlad = VLADEmbedder(
    n_clusters=256,                  # number of visual words
    kmeans_params={"rng": 0},        # forwarded to  KMeans
    pca_params={"n_components": 64}, # optional; omit for no PCA
)
vlad.learn(images)                   # fits the PCA (if any) then K-Means
```

`n_clusters` is passed directly, not inside `kmeans_params` (doing both raises a
`ValueError`). Everything else in `kmeans_params` is handed straight to the
[`KMeans`](../../pyvisim/encoders/_clustering/kmeans.py) model, and `pca_params` to the
[`PCA`](../../pyvisim/encoders/_clustering/pca.py) model. Once fitted, save with
`vlad.save_to_disk("vlad")` and reload with `VLADEmbedder.load_from_disk("vlad.embedder")`, see [base_embedder.md](base_embedder.md).

## K-Means parameters (`kmeans_params`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_init` | `1` | Number of k-means++ seedings to run; the refined codebook with the lowest distortion is kept. Raise it for better, more stable vocabularies. |
| `thresh` | `1e-05` | Stops each refinement once the change in distortion drops below this (there is no maximum-iteration count). |
| `check_finite` | `True` | Whether to validate that the input contains only finite numbers. Turn it off for a small speed-up. |
| `rng` | `None` | Seed (`int`) or `numpy.random.Generator` for reproducible fitting. |

For example, a fully reproducible embedder that keeps the best of five seedings:

```python
vlad = VLADEmbedder(n_clusters=256, kmeans_params={"rng": 0, "n_init": 5})
```

## PCA parameters (`pca_params`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_components` | (required) | Number of components to keep. Must be at most `min(n_samples, n_features)` of the training descriptors. |
| `whiten` | `False` | Scale each projected component to unit variance. Components with near-zero variance (rank-deficient descriptors) are floored at machine epsilon so the output stays finite. |
| `svd_solver` | `"auto"` | `"full"` (economy SVD), `"covariance_eigh"` (eigendecomposition of the feature covariance, fastest for many samples with few features), `"arpack"` (truncated SVD, computes only `n_components` singular triplets), or `"auto"`, which picks between them based on the training shape. |
| `tol` | `0.0` | Convergence tolerance of the `"arpack"` solver (0 means machine precision). Ignored by the other solvers. |
| `rng` | `None` | Seed (`int`) or `numpy.random.Generator` for the `"arpack"` solver's starting vector. Ignored by the other solvers. |

## How `embed` works

For each image:

1. Extract local descriptors with the feature extractor (default `RootSIFT`).
2. Apply PCA if one is set.
3. Hard-assign each descriptor to its nearest KMeans centroid.
4. For each cluster, accumulate the **residual** `descriptor - centroid`. This is the
   first-order statistic that defines VLAD.
5. Power-normalize (`sign(x) * |x|^power_norm_weight`), then L2-normalize per cluster
   row.
6. Flatten to `(K * D,)` if `flatten=True`.

A batch returns a stacked `(num_images, K * D)` array.

## References

- R. Arandjelović and A. Zisserman. "All About VLAD". In: 2013 IEEE Conference on
  Computer Vision and Pattern Recognition. 2013, pp. 1578-1585.
  doi: 10.1109/CVPR.2013.207.
