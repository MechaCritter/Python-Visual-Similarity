# ImageEncoderBase

File: [`_base_encoder.py`](../../pyvisim/encoders/_base_encoder.py)

`ImageEncoderBase` holds all logic shared by `VLADEncoder` and `FisherVectorEncoder`.
It implements `SimilarityMetric` and leaves `encode` abstract for subclasses. If you
add a new aggregation-based encoder, subclass this.

## What it manages

A concrete encoder is the combination of:

1. a **feature extractor** (`FeatureExtractorBase`),
2. an optional **PCA** model,
3. a **clustering model** (`KMeans` for VLAD, `GaussianMixture` for Fisher),
4. a **similarity function**.

The base class wires these together, validates their dimensions, and provides
`learn`, `encode` (abstract), `generate_encoding_map`, and `similarity_score`.

## Constructing an encoder

Two mutually exclusive ways to supply a clustering model:

- Pass `weights=` (a `KMeansWeights` / `GMMWeights` enum member). The base class loads
  the pickled model, and if the weight name contains `PCA` it also loads the matching
  PCA model automatically. When `weights` is given, the `clustering_model` and `pca`
  arguments are ignored.
- Pass an explicit `clustering_model=` (and optionally `pca=`) that you trained
  yourself or loaded elsewhere.
