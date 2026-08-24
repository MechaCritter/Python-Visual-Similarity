# Image Store

File: [`pyvisim/image_store/__init__.py`](../pyvisim/image_store/__init__.py)
(Implementation: [`pyvisim/image_store/image_store.py`](../pyvisim/image_store/image_store.py))

> [!NOTE]
> Requires extra: `search` (`pip install "pyvisim[search]"`)

`InMemoryImageEmbeddingStore` is the gallery object the retrieval pipeline is built
around. You give it a list of image paths, an embedder, and the kind of index you want;
it embeds every image, builds a FAISS index over the embeddings, and from then on you
search it. The store keeps only the index in memory, not a second copy of the
embeddings, so it stays lean even for big galleries.

## Building a store

```python
from pyvisim.classic import VLADEmbedder
from pyvisim.image_store import InMemoryImageEmbeddingStore

embedder = VLADEmbedder(n_clusters=64)
embedder.learn(train_images)

store = InMemoryImageEmbeddingStore(
    ["a.jpg", "b.jpg", "c.jpg"],   # embeds all three now
    embedder,
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
    ["a.jpg", "missing.jpg"], embedder, skip_errors=True
)
# FutureWarning: Skipped 1 image(s) that could not be embedded.
```

Any object that satisfies the [`Embedder`](typing.md) protocol works here, so individual
embedders and a `Pipeline` are both fair game.

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

## Search indexes

Two search structures live next to the store, both compiled into the package, so
neither needs an optional dependency. They take the gallery matrix directly and
answer batched queries:

```python
from pyvisim.image_store import BruteForceIndex, HnswIndex

index = HnswIndex(vectors, space="cosine", m=16, ef_construction=200)
scores, ids = index.search(query_vectors, k=5)   # both (M, k)
```

`ids` are row numbers into the gallery the index was built over, and `scores` are
distances, so **lower is more similar** in every space. A gallery holding fewer than
`k` vectors pads the free columns with the id `-1`.

- `HnswIndex` walks a multi-layer proximity graph, which makes it approximate but
  fast and untrained. Use it once scanning the whole gallery gets expensive.
- `BruteForceIndex` compares every query against every gallery vector. It is exact
  and needs no tuning, and it is the reference to measure a graph against.

The index owns the vectors, and hands them out read-only through `index.vectors`.
`HnswIndex` decodes them out of the graph on every access, so each one is a fresh
copy; `BruteForceIndex` hands back the matrix it holds. To change the gallery, call
`index.update(new_vectors)`, which rebuilds the structure from scratch.

### `HnswIndex` parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `space` | `"cosine"` | Metric space. `"cosine"` stores the vectors L2-normalised and scores `1 - cosine_similarity`, `"ip"` scores `1 - inner_product`, `"l2"` scores the squared Euclidean distance. |
| `m` | `16` | Bidirectional links created per node. Higher values raise recall on high-dimensional data and cost memory. |
| `ef_construction` | `200` | Size of the candidate list kept while building the graph. Higher values build a better graph, more slowly. |
| `ef_search` | `50` | Size of the candidate list kept at query time. Higher values raise recall and cost query time. A search for more than `ef_search` neighbours raises it to `k`. |
| `random_seed` | `100` | Seed of the level generator, which decides the layer each vector is inserted at. |
| `num_threads` | `-1` | Threads used to build the graph and to run batched queries. `-1` uses every available core. |

### `BruteForceIndex` parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `space` | `"cosine"` | Metric space, read exactly as above. |
| `num_threads` | `-1` | Threads used to run batched queries. `-1` uses every available core. |

## Saving and loading

`save_to_disk(path)` writes everything you need to rebuild the store, the embeddings,
the image paths, the index configuration and the fully serialised embedder, to a single
[safetensors](https://github.com/huggingface/safetensors) file (the `.safetensors`
suffix is added if you leave it off). `load_from_disk(path)` reconstructs the embedder
and re-trains the index from the saved embeddings, so no image is embedded again:

```python
store.save_to_disk("gallery.safetensors")
restored = InMemoryImageEmbeddingStore.load_from_disk("gallery.safetensors")
```
