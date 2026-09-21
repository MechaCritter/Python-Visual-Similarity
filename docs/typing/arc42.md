# arc42: typing

Software architecture of `pyvisim.typing`. This document is for developers and
is not part of the published documentation.

## Building block view

The module holds the input types and the normalization helpers that every
public method accepts, plus the protocols the library's own components are
written against. The implementation is split in two: numeric types and image
normalization live in `pyvisim/typing/numeric.py`, the embedder protocol in
`pyvisim/typing/embedders.py`.

## Architecture decisions

### Components are coupled through protocols instead of base classes

`Embedder`, `EmbeddingStore` and `SearchIndex` are `typing.Protocol` types, so
a class satisfies one by having the right methods and does not need to inherit
from it. `VLADEmbedder`, `FisherVectorEmbedder` and `Pipeline` satisfy
`Embedder` without importing it. In turn, `InMemoryImageEmbeddingStore` accepts
any of them without importing the concrete embedder classes, and `top_k_map`
and `top_k_accuracy` stay decoupled from the concrete store.

### Every input is normalized to one canonical image once per call

Whatever a caller passes is converted to a `uint8` array in `[0, 255]`, with
the axes read off the `dims` string, once per call and before the feature
extractor sees it. That single conversion point is why `dims` and
`value_range` appear on every method that takes image data, instead of each
component inventing its own layout convention.

### float32 is the storage dtype, float64 the compute dtype

Three float aliases are exported, and the choice between them is explained
below:

1. `Float32NumpyArray` is used for data that is kept in memory or handed to a
   search index: the descriptors a feature extractor returns, the embeddings
   that reach an `EmbeddingStore`, and the gallery and query matrices of every
   `SearchIndex`.
   - The extractors already produce `float32` (torch models, the vendored
     SIFT), there are enough descriptors that halving the memory footprint
     matters, and `float32` is the only dtype hnswlib and FAISS accept.
   - VLAD vectors are kept in `float32` as well, since they are sums of
     residuals, which are numerically benign, and are written straight into an
     index.
2. `Float64NumpyArray` is used for numbers that are being reduced, normalized
   or scored: the pairwise metrics in `pyvisim.distance`, the parameters of the
   GMM and the PCA, Fisher vectors, and the score matrices of the dense
   metrics.
   - The operations behind them (the dot-product expansion of the Euclidean
     distance, EM with log-sum-exp, SVD, divisions by small covariances) lose
     precision in `float32`, so their inputs are promoted to `float64` on
     entry.
   - The result is narrowed back to `float32` only where it is written into an
     index or returned as a public score matrix.
3. `FloatNumpyArray` is used wherever either dtype may arrive, for example on
   the descriptors after an optional PCA projection or on the inputs of the
   pairwise metrics.
   - There is one place where a `float64` batch is narrowed to `float32` before
     the computation, the SSIM kernel input, see `docs/structural/arc42.md`.
