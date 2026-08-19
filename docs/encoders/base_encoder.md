# ImageEmbedderBase

File: [`_base_embedder.py`](../../pyvisim/encoders/_base_embedder.py)

`ImageEmbedderBase` holds all logic shared by `VLADEmbedder` and `FisherVectorEmbedder`.
It implements `SimilarityMetric` and leaves `embed` abstract for subclasses. If you
add a new aggregation-based embedder, subclass this.

## What it manages

A concrete embedder is the combination of:

1. a **feature extractor** (`FeatureExtractorBase`),
2. an optional **PCA** model,
3. a **clustering model** (`KMeans` for VLAD, `DiagCovarGaussianMixture` for Fisher)
4. a **similarity function**.

The base class wires these together, validates their dimensions, and provides
`learn`, `to_dict`/`from_dict`, `save_to_disk`/`load_from_disk`, `embed` (abstract),
and `similarity_score`.

## Constructing an embedder

The embedder classes are constructed like this:

- `VLADEmbedder` takes `n_clusters` plus an optional `kmeans_params` dict.
- `FisherVectorEmbedder` takes `n_components` plus an optional `gmm_params` dict.
- Both take an optional `pca_params` dict (must include `n_components`) to add a PCA
  step. Leave it out and no PCA is applied.

Everything in `kmeans_params` / `gmm_params` / `pca_params` is forwarded verbatim to the
underlying `KMeans`, `DiagCovarGaussianMixture` and `PCA` models. See [vlad.md](vlad.md)
and [fisher_vector.md](fisher_vector.md) for the parameters each one accepts.

## Training and persistence

The models start unfitted, so you have to train before embedding:

- `learn(images)` extracts features from the images, fits the configured PCA first (if
  any), then fits the clustering model. Dimension checks against the feature extractor
  and PCA are deferred until the models are actually fitted.
- `save_to_disk(path)` writes the whole embedder to a versioned safetensors `.embedder`
  file: the fitted clustering model, the PCA model, the normalization hyperparameters,
  the similarity metric name and the feature-extractor configuration (the `.embedder`
  suffix is added if you leave it off). It raises `NotFittedError` if you haven't called
  `learn` yet.
- `load_from_disk(path)` rebuilds the embedder from that file, feature extractor and
  similarity metric included, so the path is all you pass. A `DeepConvFeature` using the
  default torchvision model is rebuilt from default weights; one you supplied yourself
  has its `state_dict` restored from the file.

This save/load round-trip is the supported way to reuse a trained embedder.

`to_dict`/`from_dict` expose the same state as a plain dictionary (no file involved);
`save_to_disk`/`load_from_disk` are thin wrappers over them. This is also how an
[`InMemoryImageEmbeddingStore`](../image_store.md) embeds the embedder when it serialises
a gallery.

## Indexing images by file path

To turn a folder of images into a searchable gallery, hand the paths and the fitted
embedder to an [`InMemoryImageEmbeddingStore`](../image_store.md): it embeds each image,
indexes the embeddings, and lets you search by similarity.
