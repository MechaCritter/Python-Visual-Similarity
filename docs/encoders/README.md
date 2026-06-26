# Encoders

An encoder turns an image into a single fixed-size vector you can use for
retrieval, clustering, and classification. Two families live here: the
clustering-based encoders (VLAD, Fisher Vector) that extract local descriptors and
aggregate them against a learned visual vocabulary, and `CLIPEncoder`, which runs
the image through a pretrained CLIP network end to end.

| Object | File | Aggregation model | Output size |
|--------|------|-------------------|-------------|
| [`VLADEncoder`](vlad.md) | [`vlad.py`](../../pyvisim/encoders/vlad.py) | KMeans | `K * D` |
| [`FisherVectorEncoder`](fisher_vector.md) | [`fisher_vector.py`](../../pyvisim/encoders/fisher_vector.py) | Gaussian Mixture Model | `2 * K * D + K` |
| [`CLIPEncoder`](clip.md) | [`clip.py`](../../pyvisim/encoders/clip.py) | n/a (pretrained CLIP) | model-defined (512 for `ViT-B-32`) |
| [`Pipeline`](pipeline.md) | [`pipeline.py`](../../pyvisim/encoders/pipeline.py) | n/a (composes encoders) | sum of members |

where `K` is the number of clusters and `D` is the local descriptor dimension.

Shared machinery lives in [`ImageEncoderBase`](base_encoder.md). The clustering
encoders build their aggregation model from the
[`pyvisim.clustering`](../clustering/README.md) package using the parameters you
pass at construction, then fit it in `learn`. Trained encoders are saved and
restored with `save_to_disk` / `load_from_disk`; the older pretrained-weight enums
in [weights.md](weights.md) still work but are deprecated. `CLIPEncoder` skips the
`learn` step entirely (its weights are already pretrained) and needs the `nn`
extra: `pip install "pyvisim[nn]"`.

## VLAD vs Fisher Vector

Both follow the same extract → aggregate → normalize flow and share the same base
class. They differ in what statistics they capture:

- **VLAD** records only first-order statistics: the sum of residuals (descriptor minus
  centroid) per cluster. Clustering is hard-assignment via KMeans.
- **Fisher Vector** records first- and second-order statistics as gradients of the GMM
  log-likelihood with respect to its weights, means, and variances. Assignment is soft
  (posterior probabilities). This makes Fisher vectors larger but more expressive.
