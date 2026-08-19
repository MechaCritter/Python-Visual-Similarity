# Classic embedders

Before deep learning took over image retrieval, this is how you turned a photo into a
vector. You ran a hand-designed local feature detector over the image, got a few
hundred descriptors out of it (one per keypoint, and the count changed from image to
image), and then aggregated that variable-length pile into a single fixed-size vector
against a vocabulary learned from your dataset. That last step is what VLAD and Fisher
Vector do, and it's what made the descriptors usable for indexing, clustering and
linear classifiers.

These methods ran the show in the late 2000s and early 2010s, and they won image
retrieval benchmarks right up until CNN features displaced them. They're still worth
having around: they train in minutes on a laptop with no GPU, the vocabulary learns
from *your* images rather than someone else's dataset, and you can read every step of
the math. If you want end-to-end learned embeddings instead, reach for `ClipEmbedder`
or the Siamese networks in [`pyvisim.neural_networks`](../neural_networks/README.md).

| Object | File | Aggregation model | Output size |
|--------|------|-------------------|-------------|
| [`VLADEmbedder`](vlad.md) | [`vlad.py`](../../pyvisim/classic/vlad.py) | KMeans | `K * D` |
| [`FisherVectorEmbedder`](fisher_vector.md) | [`fisher_vector.py`](../../pyvisim/classic/fisher_vector.py) | Gaussian Mixture Model | `2 * K * D + K` |
| [`Pipeline`](pipeline.md) | [`pipeline.py`](../../pyvisim/classic/pipeline.py) | n/a (composes embedders) | sum of members |

where `K` is the number of clusters and `D` is the local descriptor dimension.

Both embedders follow the same three-step flow:

1. **Extract** local descriptors from the image (`RootSIFT` by default).
2. **Aggregate** them against a vocabulary learned from your images, optionally after a
   [`PCA`](pca.md) step that shrinks the descriptors first.
3. **Normalize** the result so cosine similarity behaves.

Shared machinery lives in [`ImageEmbedderBase`](image_embedder_base.md). The clustering
models (`KMeans`, `DiagCovarGaussianMixture` and `PCA`) live inside the package at
`pyvisim/classic/_clustering/`; you configure them with the parameters you pass at
construction, and `learn` fits them. Trained embedders are saved and restored with
`save_to_disk` / `load_from_disk`.

## VLAD vs Fisher Vector

Both share the same base class and the same flow. They differ in what statistics they
keep:

- **VLAD** records only first-order statistics: the sum of residuals (descriptor minus
  centroid) per cluster. Assignment is hard, via KMeans. Smaller and faster.
- **Fisher Vector** records first- and second-order statistics as gradients of the GMM
  log-likelihood with respect to its weights, means and variances. Assignment is soft
  (posterior probabilities). Bigger vectors, but more expressive.

Rule of thumb: start with VLAD because it's cheaper to fit and to store, and move to
Fisher Vector if you need the extra accuracy and can afford vectors roughly twice the
size.
