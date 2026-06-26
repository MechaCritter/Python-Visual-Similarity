# Changelog

## [Unreleased]

### Added
- `CLIPEncoder` (in `pyvisim.encoders`): a pretrained-CLIP image encoder built on
  open_clip. It maps an image straight to a CLIP embedding, so there's no feature
  extractor, clustering model, or `learn` step. Embeddings are L2-normalized by default,
  and it plugs into the usual `similarity_score` / `save_to_disk` / `load_from_disk`
  machinery.

  ```python
  from pyvisim.encoders import CLIPEncoder

  clip = CLIPEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
  embeddings = clip.encode(images)
  ```

  Saving stores only the model identifiers (`model_name`, `pretrained` tag, etc.), not
  the weights, so `.encoder` files stay tiny and open_clip re-fetches the weights on load.
- `nn` optional extra (`pip install "pyvisim[nn]"`) that pulls in `open_clip_torch` for
  the neural-network encoders. open_clip is imported lazily, so importing `pyvisim` never
  requires it; you only hit the error (with an install hint) when you build a
  `CLIPEncoder` without it installed.

## [0.6.0] - 2026-06-20

### Added
- `InMemoryImageEmbeddingStore` (in `pyvisim.image_store`): the new gallery object.
  Give it image paths, an encoder, and an index type, and it encodes everything,
  builds a FAISS index, and searches itself:

  ```python
  from pyvisim.image_store import InMemoryImageEmbeddingStore

  store = InMemoryImageEmbeddingStore(
      gallery_paths, encoder, "ivf-flat",
      quantizer="inner_product", index_params={"nlist": 100, "nprobe": 8},
  )
  results = store.retrieve_top_k_similar(query_images, k=5)
  ```

  It saves to a single `.safetensors` file (embeddings, paths, index config and the
  fully serialised encoder) and `load_from_disk` rebuilds it without re-encoding.
- `index_type` strings select the index structure: `"ivf-flat"` and `"ivf-pq"` work
  today; `"hnsw"` and `"int8"` are sketched for a future release and raise
  `NotImplementedError` for now.
- Encoders and `Pipeline` gained `to_dict`/`from_dict`, and there's a new
  `EmbeddingStore` protocol in `pyvisim.typing`.

### Changed
- `retrieve_top_k_similar(query_images, store, k=5)` now takes a store and searches
  through its index. `top_k_map` and `top_k_accuracy` take a store too, instead of a
  separate `(encoding_map, encoder)` pair.
- `ImageRetriever` now wraps a store: `ImageRetriever(store)`.
- The image indexes take the gallery as `(paths, vectors)` rather than a mapping, and
  the trained FAISS index is now the single owner of the vectors. Read them back with
  `index.reconstruct()` (or `store.embeddings`) instead of keeping a second copy.

### Breaking
- ⚠️ `ImageEncodingMap` is gone. Build an `InMemoryImageEmbeddingStore` from your image
  paths instead of a `{path: vector}` mapping.
- ⚠️ `Encoder.generate_encoding_map(...)` and `Pipeline.generate_encoding_map(...)` are
  removed. Pass the paths straight to `InMemoryImageEmbeddingStore`.
- ⚠️ `retrieve_top_k_similar` dropped its `dataset`/`encoder`/`index` arguments (and the
  brute-force path); pass a store. The same applies to `top_k_map`/`top_k_accuracy`.

## [v0.5.1] - 2026-06-19

## Fixed
- The method `_from_config` of `DeepConvFeature` was using the deprecated
  `model` argument instead of `backbone`. This version only fixed that.

## [v0.5.0] - 2026-06-19

### Added
- New `pyvisim.retrieval` package for fast similarity search. Wrap an
  `ImageEncodingMap` in an index (`ImageIndexIVFFlat` or `ImageIndexIVFPQ`,
  both `l2` or `inner_product`), then hand it to an `ImageRetriever`:

  ```python
  from pyvisim.retrieval import ImageIndexIVFFlat, ImageRetriever

  index = ImageIndexIVFFlat(encoding_map, quantizer="inner_product", nlist=100)
  retriever = ImageRetriever(index)
  results = retriever.retrieve_top_k_similar(query_images, k=5)
  ```
- New `pyvisim.functional` module holding `retrieve_top_k_similar` and the
  `Candidate(path, score)` result type.

### Changed
- `retrieve_top_k_similar` now ranks a whole batch of query images in one shot
  and returns one ranked `list[Candidate]` per query (in input order), so a
  single call can search many images at once. Pass an `index=` to run the search
  through FAISS instead of brute-force cosine.

### Breaking
- ⚠️ `retrieve_top_k_similar` moved out of `pyvisim.eval` into
  `pyvisim.functional`. Update your imports:
  `from pyvisim.functional import retrieve_top_k_similar`.
- ⚠️ Its return type changed from `list[tuple[str, float]]` to
  `list[list[Candidate]]` (one list per query image). Read `candidate.path` and
  `candidate.score` instead of unpacking a tuple.
- ⚠️ The getters for the `pca` and `clustering_model` attributes of all encoders
   are removed (the attributes are now read-only). This is in order to discourage
   users from mutating the clustering internals, which could break the
   algorithm completely. Also, once trained, there's not really a reason to
   have to mutate those models at all because any different model would be
   basically wrong for the trained encoder.
- ⚠️ The `ImageEncodingMap` does not take the `Encoder` as an argument anymore.

## [v0.4.1] - 2026-06-18

### Added
- `DeepConvFeature` now takes a `backbone` argument. Pass `"vgg16"` to grab a
  torchvision VGG16 with ImageNet weights, or hand it your own `torch.nn.Module`.
  Leave it out and you still get the default VGG16.

### Deprecated
- The `model` argument of `DeepConvFeature` is deprecated; use `backbone`
  instead. If you still pass `model`, it's used as the backbone and you'll get a
  `DeprecationWarning`. It'll be removed in a future release.

## [v0.4.0] - 2026-06-18

### Added
- `from_pretrained()` on `VLADEncoder` and `FisherVectorEncoder`, plus the
  `PretrainedVLAD` and `PretrainedFisher` enums. Pick a bundled encoder and
  you're ready to go: `VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)`.

### Changed
- Encoders now serialize to a single safetensors `.encoder` file that captures
  everything: the clustering model, PCA, normalization settings, the feature
  extractor and the similarity metric. `load_from_disk()` takes just the path
  and rebuilds the whole encoder, so there's nothing else to pass back in.
- For a `DeepConvFeature` extractor, the default torchvision model is rebuilt on
  load (only a flag is stored), while a model you supply yourself has its full
  `state_dict` embedded so your trained weights come back exactly.
- `similarity_func` is now chosen by name: `"cosine"` (default), `"euclidean"`,
  `"l1"` or `"manhattan"`.
- The pretrained Oxford-102 weights ship as `.encoder` files instead of `.pkl`,
  shrinking them from ~144 MB to ~12 MB (the K-Means training `labels_` array is
  no longer stored).

### Removed
- Dropped `joblib` entirely in favor of safetensors.
- ⚠️ You can no longer pass your own similarity function; use one of the four
  built-in metric names above.

### Deprecated
- Loading pretrained weights via `KMeansWeights`/`GMMWeights` (the `weights=`
  argument) is deprecated and will be removed in 1.0.0. Use `from_pretrained()`
  or `load_from_disk()` with `.encoder` files instead.

## [v0.3.1] - 2026-06-18

### Changed
- `ImageEncodingMap` now encodes every image up front instead of lazily on first access, which drops the in-memory buffer machinery and simplifies the class.
- `ImageEncodingMap.save_to_disk()` / `load_from_disk()` now use the safetensors format instead of HDF5. Files default to the `.safetensors` extension.
- `skip_errors` moved from `save_to_disk()` to the `ImageEncodingMap` constructor, since encoding now happens at construction time.

### Removed
- Dropped `h5py` as a dependency; added `safetensors`.
- Removed `ImageEncodingMap.clear_buffer()` (there's no buffer to clear anymore).

### Breaking
- ⚠️ Unreadable or missing images now raise (`FileNotFoundError` / `ValueError`) when the map is built, not on first access. Use `skip_errors=True` to drop them with a warning instead.
- ⚠️ Encoding maps saved with `0.3.0` (HDF5) can't be loaded by `0.3.1`; re-save them as safetensors.

## [v0.3.0] - 2026-06-17

### Added
- New encoding map feature for the encoders (#36).
- PyPI publishing step in the CI workflow, so releases ship automatically (#37).

### Changed
- Batches now fetch dynamically from PyPI instead of being hardcoded (#38).
- Moved the notebooks into a tidier layout (#35).

### Fixed
- Dropped the deprecated `project.license` TOML table in `pyproject.toml` (#39).

## [v0.2.0] - 2026-06-16

### Added
- Clustering models with a fresh public API (#19), plus docs to match (#21).
- Public types `ImageInput` and `MatLike`, and you can now pass torch images straight in (#23).
- Unit tests across the board (#24), including behavioral tests that check VLAD and Fisher Vector encoders return the same vector before and after serialization (#32).
- Early sketch of a Siamese neural net (#26).

### Changed
- Migrated tooling to `uv` (#5).
- Added ruff, pre-commit hooks, and a CI pipeline that runs on every PR (#9).
- Integrated mypy and cleaned up the type errors (#8, #13).
- Now compatible with Python 3.10 through 3.12 (#14, #15).
- Refreshed the outdated getting-started notebook (#30).

### Fixed
- `VLADEncoder` now raises if no descriptor is extracted, instead of failing silently (#11).
- Use `flatten()` instead of `squeeze()` for setid arrays, so single-element arrays behave (#29).

## [v0.1.3-alpha] - 2025-01-24

- Initial alpha release.
