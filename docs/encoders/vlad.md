# VLADEncoder

File: [`vlad.py`](../../pyvisim/encoders/vlad.py)

VLAD (Vector of Locally Aggregated Descriptors) encodes an image into a vector of
shape `(K * D,)`, where `K` is the number of KMeans clusters and `D` is the local
descriptor dimension (after optional PCA).

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
