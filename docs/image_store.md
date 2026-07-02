# pyvisim.image_store

File: [`pyvisim/image_store/__init__.py`](../pyvisim/image_store/__init__.py)
(Implementation: [`pyvisim/image_store/image_store.py`](../pyvisim/image_store/image_store.py))

> [!NOTE]
> Requires extra: `search` (`pip install "pyvisim[search]"`)

`InMemoryImageEmbeddingStore` is the gallery object the retrieval pipeline is built
around. You give it a list of image paths, an encoder, and the kind of index you want;
it encodes every image, builds a FAISS index over the embeddings, and from then on you
search it. The store keeps only the index in memory, not a second copy of the
embeddings, so it stays lean even for big galleries.

## Building a store

```python
from pyvisim.encoders import VLADEncoder
from pyvisim.image_store import InMemoryImageEmbeddingStore

encoder = VLADEncoder(n_clusters=64)
encoder.learn(train_images)

store = InMemoryImageEmbeddingStore(
    ["a.jpg", "b.jpg", "c.jpg"],   # encodes all three now
    encoder,
    "ivf-flat",                    # index structure (the default)
    quantizer="inner_product",     # rank by cosine similarity
    index_params={"nlist": 100, "nprobe": 8},
)
```

`index_type` picks the index structure:

- `"ivf-flat"` keeps the full vectors, so it's exact within the cells it scans.
- `"ivf-pq"` compresses them with product quantization for a much smaller footprint.
- `"hnsw"` and `"int8"` are sketched for upcoming releases and raise
  `NotImplementedError` for now.

Anything in `index_params` is forwarded straight to the chosen index (e.g. `nlist`,
`nprobe`, and for IVF-PQ `m` and `nbits`). See the [retrieval docs](retrieval/README.md)
for what each index accepts.

A missing file raises `FileNotFoundError` and a file that isn't a valid image raises
`ValueError`. If you'd rather skip the bad ones, pass `skip_errors=True` and they're
dropped with a warning instead:

```python
store = InMemoryImageEmbeddingStore(
    ["a.jpg", "missing.jpg"], encoder, skip_errors=True
)
# FutureWarning: Skipped 1 image(s) that could not be encoded.
```

Any object that satisfies the [`Encoder`](typing.md) protocol works here, so individual
encoders and a `Pipeline` are both fair game.

## Searching

The store searches itself, so you don't need a separate retriever:

```python
results = store.retrieve_top_k_similar([query_a, query_b], k=5)
for ranked in results:        # one list per query image, in input order
    for candidate in ranked:  # already sorted, best match first
        print(candidate.path, candidate.score)
```

You can also reach the embeddings (reconstructed from the index on demand) and the raw
FAISS `search`:

```python
store.paths        # gallery image paths, in index order
store.embeddings   # the (N, D) matrix, read back from the index
store.index        # the underlying pyvisim.retrieval.ImageIndex
```

For an inner-product store the embeddings come back L2-normalised (the form they were
indexed in); for an IVF-PQ store they're the decompressed approximation.

## Saving and loading

`save_to_disk(path)` writes everything you need to rebuild the store, the embeddings,
the image paths, the index configuration and the fully serialised encoder, to a single
[safetensors](https://github.com/huggingface/safetensors) file (the `.safetensors`
suffix is added if you leave it off). `load_from_disk(path)` reconstructs the encoder
and re-trains the index from the saved embeddings, so no image is encoded again:

```python
store.save_to_disk("gallery.safetensors")
restored = InMemoryImageEmbeddingStore.load_from_disk("gallery.safetensors")
```
