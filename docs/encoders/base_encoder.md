# ImageEncoderBase

File: [`_base_encoder.py`](../../pyvisim/encoders/_base_encoder.py)

`ImageEncoderBase` holds all logic shared by `VLADEncoder` and `FisherVectorEncoder`.
It implements `SimilarityMetric` and leaves `encode` abstract for subclasses. If you
add a new aggregation-based encoder, subclass this.

## What it manages

A concrete encoder is the combination of:

1. a **feature extractor** (`FeatureExtractorBase`),
2. an optional **PCA** model,
3. a **clustering model** (`KMeans` for VLAD, `GaussianMixtureModel` for Fisher; both
   from [`pyvisim.clustering`](../clustering/README.md)),
4. a **similarity function**.

The base class wires these together, validates their dimensions, and provides
`learn`, `save_to_disk`/`load_from_disk`, `encode` (abstract), `generate_encoding_map`,
and `similarity_score`.

## Constructing an encoder

The encoder classes are constructed like this:

- `VLADEncoder` takes `n_clusters` plus an optional `kmeans_params` dict.
- `FisherVectorEncoder` takes `n_components` plus an optional `gmm_params` dict.
- Both take an optional `pca_params` dict (must include `n_components`) to add a PCA
  step. Leave it out and no PCA is applied.

Everything in `kmeans_params` / `gmm_params` / `pca_params` is forwarded verbatim to the
underlying scikit-learn models (see scikit-learn for `KMeans` and `GaussianMixture` documentation). See
[vlad.md](vlad.md) and [fisher_vector.md](fisher_vector.md) for the per-encoder details.

## Training and persistence

The models start unfitted, so you have to train before encoding:

- `learn(images)` extracts features from the images, fits the configured PCA first (if
  any), then fits the clustering model. Dimension checks against the feature extractor
  and PCA are deferred until the models are actually fitted.
- `save_to_disk(path)` writes the fitted clustering model, the PCA model, and the
  normalization hyperparameters to a versioned `.encoder` file (the `.encoder` suffix is
  added if you leave it off). It raises `NotFittedError` if you haven't called `learn`
  yet.
- `load_from_disk(path)` rebuilds the encoder from that file. The feature extractor and
  similarity function aren't serialized, so you pass them again here (the feature
  extractor defaults to `RootSIFT`); its output dimension has to match the saved PCA or
  clustering model.

This save/load round-trip is the supported way to reuse a trained encoder. The old
`weights=` enum path still works but is deprecated, see [weights.md](weights.md).

## Encoding images by file path

`generate_encoding_map(image_paths)` returns an
[`ImageEncodingMap`](../image_store.md): a lazy `{path: encoding}` mapping that reads and
encodes each image the first time you access it, then keeps the vector in memory. It
behaves like the dict it used to return (index by path, iterate, `len`, `.values()`), and the whole mapping can be persisted to an HDF5 file.
