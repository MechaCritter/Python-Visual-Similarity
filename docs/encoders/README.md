# Encoders

An encoder turns an image into a single fixed-size vector by extracting local
descriptors and aggregating them against a learned visual vocabulary. The resulting
vectors are used for retrieval, clustering, and classification.

| Object | File | Aggregation model | Output size |
|--------|------|-------------------|-------------|
| [`VLADEncoder`](vlad.md) | [`vlad.py`](../../pyvisim/encoders/vlad.py) | KMeans | `K * D` |
| [`FisherVectorEncoder`](fisher_vector.md) | [`fisher_vector.py`](../../pyvisim/encoders/fisher_vector.py) | Gaussian Mixture Model | `2 * K * D + K` |
| [`Pipeline`](pipeline.md) | [`pipeline.py`](../../pyvisim/encoders/pipeline.py) | n/a (composes encoders) | sum of members |

where `K` is the number of clusters and `D` is the local descriptor dimension.

Shared machinery lives in [`ImageEncoderBase`](base_encoder.md), and pretrained
clustering/PCA models are exposed through the enums documented in
[weights.md](weights.md).

## VLAD vs Fisher Vector

Both follow the same extract → aggregate → normalize flow and share the same base
class. They differ in what statistics they capture:

- **VLAD** records only first-order statistics: the sum of residuals (descriptor minus
  centroid) per cluster. Clustering is hard-assignment via KMeans.
- **Fisher Vector** records first- and second-order statistics as gradients of the GMM
  log-likelihood with respect to its weights, means, and variances. Assignment is soft
  (posterior probabilities). This makes Fisher vectors larger but more expressive.
