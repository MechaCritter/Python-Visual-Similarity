# VLADEncoder

File: [`vlad.py`](../../pyvisim/encoders/vlad.py)

VLAD (Vector of Locally Aggregated Descriptors) encodes an image into a vector of
shape `(K * D,)`, where `K` is the number of KMeans clusters and `D` is the local
descriptor dimension (after optional PCA).

## Constructing one

VLAD always clusters with K-Means, so you configure that model through the encoder:

```python
from pyvisim.encoders import VLADEncoder

vlad = VLADEncoder(
    n_clusters=256,                    # number of visual words
    kmeans_params={"random_state": 0}, # forwarded to sklearn.cluster.KMeans
    pca_params={"n_components": 64},   # optional; omit for no PCA
)
vlad.learn(images)                     # fits the PCA (if any) then K-Means
```

`n_clusters` is passed directly, not inside `kmeans_params` (doing both raises a
`ValueError`). Everything else in `kmeans_params` / `pca_params` is handed straight to
the matching scikit-learn estimator. Once fitted, save with `vlad.save_to_disk("vlad")`
and reload with `VLADEncoder.load_from_disk("vlad.encoder")`, see
[base_encoder.md](base_encoder.md).

## How `encode` works

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
