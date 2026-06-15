# Developer Overview

`pyvisim` computes image similarity by turning images into fixed-size vectors and
comparing those vectors with a similarity function. This document explains how the
pieces fit together so you can extend the library or debug it.

## Package layout

```
pyvisim/
├── _base_classes.py     Abstract interfaces (SimilarityMetric, FeatureExtractorBase)
├── _config.py           Paths and logging setup
├── _utils.py            cosine_similarity, image IO, clustering + plotting helpers
├── _errors.py           Custom exceptions
├── typing/              Public types and helper methods
├── eval.py              Retrieval metrics (top-k, mAP, accuracy)
├── encoders/            VLAD, Fisher Vector, Pipeline, pretrained weights
├── clustering/          KMeans, GaussianMixtureModel, PCA
├── features/            SIFT, RootSIFT, DeepConvFeature, Lambda
├── datasets/            OxfordFlowerDataset
└── neural_networks/     Siamese network (planned, not yet implemented)
```

Per-area docs:

- [Typing](typing.md): Public types (`MatLike`, `ImageInput`).
- [Encoders](encoders/): how images become vectors.
- [Clustering](clustering/): the KMeans, GMM, and PCA models the encoders build their
  vocabulary with.
- [Features](features/): how local descriptors are extracted from an image.
- [Neural networks](neural_networks/): planned Siamese network.
- [Dataset](dataset/): the bundled Oxford Flowers dataset class.

## The core pipeline

Every encoder follows the same three-stage flow:

```
image (NumPy array)
   │  feature extractor   (features/)
   ▼
local descriptors  (N, D)
   │  optional PCA, then clustering model (KMeans or GMM)
   ▼
aggregated vector  (fixed size, independent of N)
   │  power + L2 normalization
   ▼
encoding  →  similarity_func  →  similarity score
```

The key property is that the number of local descriptors `N` varies per image, but
the aggregated encoding has a fixed length. That is what makes two images comparable
with a single similarity function.

## Two abstract interfaces

Everything is built on the two abstract base classes in
[`_base_classes.py`](../pyvisim/_base_classes.py):

- `SimilarityMetric`: anything that can produce a `similarity_score` between two
  images. Both the encoders and the `Pipeline` implement this.
- `FeatureExtractorBase`: anything that maps one image to a `(N, D)` array of
  descriptors via `__call__`, and reports its `output_dim`. Used to validate that a
  feature extractor matches the clustering model and PCA it is paired with.

## Design decisions worth knowing

- **NumPy, torch, or array-like images.** Encoders and feature extractors accept `MatLike`
  inputs: NumPy arrays, PyTorch tensors, or anything `numpy.asarray` can turn into a
  numeric array. Use the `dims` string (e.g. `"HWC"`, `"BCHW"`) to describe the axis
  layout, and `value_range` to rescale float inputs into `[0, 255]`. See
  [typing.md](typing.md).
- **Similarity function is pluggable and guarded.** Any callable taking two `(N, D)`
  and `(M, D)` arrays and returning an `(N, M)` matrix can be used. On assignment it
  is probed with dummy input; if it does not return the expected shape,
  fall back to a row-by-row loop. Default is cosine similarity.
- **Trained encoders persist to `.encoder` files.** `save_to_disk` / `load_from_disk`
  serialize the fitted clustering model, PCA, and normalization settings. This replaces
  the deprecated `KMeansWeights` / `GMMWeights` enum loading path, which still works for
  now but warns. See [encoders/weights.md](encoders/weights.md).

## Evaluation

[`eval.py`](../pyvisim/eval.py) provides methods to compute the performance
of the retrieval pipeline, given the ground-truth labels. Currently available:
- `retrieve_top_k_similar`: return the top-k most similar items to a query.
- `top_k_map`: mean Average Precision over a set of queries.
- `top_k_accuracy`: fraction of queries whose top-k contains a label match.
