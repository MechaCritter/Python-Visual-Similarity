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
`learn`, `to_dict`/`from_dict`, `save_to_disk`/`load_from_disk`, `encode` (abstract),
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
- `save_to_disk(path)` writes the whole encoder to a versioned safetensors `.encoder`
  file: the fitted clustering model, the PCA model, the normalization hyperparameters,
  the similarity metric name and the feature-extractor configuration (the `.encoder`
  suffix is added if you leave it off). It raises `NotFittedError` if you haven't called
  `learn` yet.
- `load_from_disk(path)` rebuilds the encoder from that file, feature extractor and
  similarity metric included, so the path is all you pass. A `DeepConvFeature` using the
  default torchvision model is rebuilt from default weights; one you supplied yourself
  has its `state_dict` restored from the file.
- `from_pretrained(enum)` loads one of the bundled pretrained encoders, see
  [weights.md](weights.md).

This save/load round-trip is the supported way to reuse a trained encoder. The old
`weights=` enum path still works but is deprecated, see [weights.md](weights.md).

`to_dict`/`from_dict` expose the same state as a plain dictionary (no file involved);
`save_to_disk`/`load_from_disk` are thin wrappers over them. This is also how an
[`InMemoryImageEmbeddingStore`](../image_store.md) embeds the encoder when it serialises
a gallery.

## Indexing images by file path

To turn a folder of images into a searchable gallery, hand the paths and the fitted
encoder to an [`InMemoryImageEmbeddingStore`](../image_store.md): it encodes each image,
indexes the embeddings, and lets you search by similarity.
