# VLADEmbedder

File: [`vlad.py`](../../pyvisim/classic/vlad.py)

VLAD (Vector of Locally Aggregated Descriptors) was introduced by Jegou et al. in 2010
as a compact alternative to bag-of-words for large-scale image retrieval, back when
hand-designed local features were still the state of the art. It embeds an image into a
vector of shape `(K * D,)`, where `K` is the number of KMeans clusters and `D` is the
local descriptor dimension (after optional PCA).

The idea is simple enough to hold in your head: build a visual vocabulary with K-Means,
then, instead of just counting how many descriptors landed in each cluster, record *how
far off-centre* they landed. Those residuals keep far more information than a histogram
of counts, which is what made VLAD competitive at a fraction of the size.

## Constructing one

VLAD always clusters with K-Means, so you configure that model through the embedder:

```python
from pyvisim.classic import VLADEmbedder

vlad = VLADEmbedder(
    n_clusters=256,                  # number of visual words
    kmeans_params={"rng": 0},        # forwarded to  KMeans
    pca_params={"n_components": 64}, # optional; omit for no PCA
)
vlad.learn(images)                   # fits the PCA (if any) then K-Means
```

`n_clusters` is passed directly, not inside `kmeans_params` (doing both raises a
`ValueError`). Everything else in `kmeans_params` is handed straight to the
[`KMeans`](../../pyvisim/classic/_clustering/kmeans.py) model, and `pca_params` to the
[`PCA`](../../pyvisim/classic/_clustering/pca.py) model. Once fitted, save with
`vlad.save_to_disk("vlad")` and reload with `VLADEmbedder.load_from_disk("vlad.embedder")`, see
[image_embedder_base.md](image_embedder_base.md).

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

See [pca.md](pca.md).

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
