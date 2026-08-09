# Encoders

An encoder turns an image into a single fixed-size vector you can use for
retrieval, clustering, and classification. The encoders here are the
clustering-based ones (VLAD, Fisher Vector) that extract local descriptors and
aggregate them against a learned visual vocabulary. If you want end-to-end CLIP
embeddings instead, reach for `ClipEmbedder` in
[`pyvisim.neural_networks`](../neural_networks/README.md).

| Object | File | Aggregation model | Output size |
|--------|------|-------------------|-------------|
| [`VLADEncoder`](vlad.md) | [`vlad.py`](../../pyvisim/encoders/vlad.py) | KMeans | `K * D` |
| [`FisherVectorEncoder`](fisher_vector.md) | [`fisher_vector.py`](../../pyvisim/encoders/fisher_vector.py) | Gaussian Mixture Model | `2 * K * D + K` |
| [`Pipeline`](pipeline.md) | [`pipeline.py`](../../pyvisim/encoders/pipeline.py) | n/a (composes encoders) | sum of members |

where `K` is the number of clusters and `D` is the local descriptor dimension.

Shared machinery lives in [`ImageEncoderBase`](base_encoder.md). The clustering
encoders build their aggregation model from the `KMeans`, `GaussianMixtureModel`
and `PCA` classes bundled inside the encoders package
(`pyvisim/encoders/_clustering/`) using the parameters you pass at construction, then
fit it in `learn`. Trained encoders are saved and
restored with `save_to_disk` / `load_from_disk`.

## VLAD vs Fisher Vector

Both follow the same extract → aggregate → normalize flow and share the same base
class. They differ in what statistics they capture:

- **VLAD** records only first-order statistics: the sum of residuals (descriptor minus
  centroid) per cluster. Clustering is hard-assignment via KMeans.
- **Fisher Vector** records first- and second-order statistics as gradients of the GMM
  log-likelihood with respect to its weights, means, and variances. Assignment is soft
  (posterior probabilities). This makes Fisher vectors larger but more expressive.
