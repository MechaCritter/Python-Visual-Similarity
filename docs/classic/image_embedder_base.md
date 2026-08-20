# ImageEmbedderBase and friends

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
