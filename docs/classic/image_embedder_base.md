# ImageEmbedderBase and friends

File: [`_base_embedder.py`](../../pyvisim/classic/_base_embedder.py)

The classic embedders sit on a short chain of base classes:

- `ImageEmbedderBase` (in [`pyvisim/_base_classes.py`](../../pyvisim/_base_classes.py))
  is the contract every embedder in pyvisim satisfies, classic or neural: an `embed`
  method, a named `similarity_func`, and a `similarity_score` built on top of the two.
- `SerializableImageEmbedder` adds persistence: `to_dict`/`from_dict` plus
  `save_to_disk`/`load_from_disk`. Neural embedders skip this and use torch checkpoints
  instead.
- `FeatureBasedEmbedder` adds a feature extractor.
- `ClusteringBasedEmbedder` adds the clustering model, the PCA step and `learn`.

`VLADEmbedder` and `FisherVectorEmbedder` are thin layers on top of that last one, so
if you're adding another aggregation-based embedder, that's where to subclass.

## What a concrete embedder is made of

1. a **feature extractor** (`FeatureExtractorBase`),
2. an optional **PCA** model,
3. a **clustering model** (`KMeans` for VLAD, `DiagCovarGaussianMixture` for Fisher),
4. a **similarity function**.

The base classes wire these together and validate that their dimensions line up, so you
find out about a mismatched feature extractor when you set it, not halfway through a
training run.

## Constructing an embedder

- `VLADEmbedder` takes `n_clusters` plus an optional `kmeans_params` dict.
- `FisherVectorEmbedder` takes `n_components` plus an optional `gmm_params` dict.
- Both take an optional `pca_params` dict (must include `n_components`) to add a PCA
  step. Leave it out and no PCA is applied.

Everything in `kmeans_params` / `gmm_params` / `pca_params` is forwarded verbatim to the
underlying `KMeans`, `DiagCovarGaussianMixture` and `PCA` models. See [vlad.md](vlad.md),
[fisher_vector.md](fisher_vector.md) and [pca.md](pca.md) for the parameters each one
accepts.

## Training and persistence

The models start unfitted, so train before you embed:

- `learn(images)` extracts features from the images, fits the configured PCA first (if
  any), then fits the clustering model. Dimension checks against the feature extractor
  and PCA wait until the models are actually fitted.
- `save_to_disk(path)` writes the whole embedder to a versioned safetensors `.embedder`
  file: the fitted clustering model, the PCA model, the normalization hyperparameters,
  the similarity metric name and the feature-extractor configuration. The `.embedder`
  suffix is added if you leave it off. Raises `NotFittedError` if you haven't called
  `learn` yet.
- `load_from_disk(path)` rebuilds the embedder from that file, feature extractor and
  similarity metric included, so the path is all you pass. A `DeepConvFeature` using the
  default torchvision model is rebuilt from default weights; one you supplied yourself
  has its `state_dict` restored from the file.

That round-trip is the supported way to reuse a trained embedder. Loading a file saved
by a different embedder class raises, so you can't hand a Fisher file to
`VLADEmbedder.load_from_disk`.

`to_dict`/`from_dict` expose the same state as a plain dictionary with no file involved;
`save_to_disk`/`load_from_disk` are thin wrappers over them. That's also how an
[`InMemoryImageEmbeddingStore`](../image_store.md) tucks the embedder in when it
serialises a gallery.

## Indexing images by file path

To turn a folder of images into a searchable gallery, hand the paths and the fitted
embedder to an [`InMemoryImageEmbeddingStore`](../image_store.md): it embeds each image,
indexes the embeddings, and lets you search by similarity.
