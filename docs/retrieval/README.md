# pyvisim.retrieval

File: [`pyvisim/retrieval/__init__.py`](../../pyvisim/retrieval/__init__.py)

The `retrieval` package finds the images in a gallery that look most like a query
image. The gallery is an
[`InMemoryImageEmbeddingStore`](../image_store.md), which builds a FAISS index over
the embeddings so the search is fast.

## Quick start

The store searches itself, so most of the time you just call it directly:

```python
from pyvisim.image_store import InMemoryImageEmbeddingStore

store = InMemoryImageEmbeddingStore(
    gallery_paths, encoder, "ivf-flat",
    quantizer="inner_product", index_params={"nlist": 100, "nprobe": 8},
)

results = store.retrieve_top_k_similar([query_a, query_b], k=5)
for ranked in results:                 # one list per query image, in input order
    for candidate in ranked:           # already sorted, best match first
        print(candidate.path, candidate.score)
```

You pass the query image array(s), and you get back one ranked `list[Candidate]`
per query. Each `Candidate` is a small named tuple of `(path, score)`.

If you prefer a façade object, `ImageRetriever` wraps a store and forwards to the same
method:

```python
from pyvisim.retrieval import ImageRetriever

retriever = ImageRetriever(store)
results = retriever.retrieve_top_k_similar([query_a, query_b], k=5)
```

## Picking a quantizer

Both indexes take `quantizer="l2"` or `quantizer="inner_product"`:

- `"l2"` ranks by Euclidean distance, so lower scores are better.
- `"inner_product"` L2-normalizes the vectors first, so it ranks by cosine
  similarity and higher scores are better. This is usually what you want for
  similarity search.

## Choosing an index

- `ImageIndexIVFFlat` keeps the full gallery vectors, so it's exact within the
  cells it scans. Tune it with `nlist` (number of Voronoi cells) and `nprobe`
  (cells scanned per query); a higher `nprobe` trades speed for recall.
- `ImageIndexIVFPQ` adds product-quantization on top, compressing each vector
  into `m` sub-vectors of `nbits` bits each. It uses far less memory at the cost
  of approximate distances. `m` must divide the vector dimensionality, and the
  gallery needs at least `2 ** nbits` vectors to train the codebook.

Both expose their coarse-quantizer centroids via `index.cluster_centers`. Two more
structures, `ImageIndexHNSW` (the `"hnsw"` store) and `ImageIndexScalarQuantizer`
(the `"int8"` store), are sketched for upcoming releases and raise
`NotImplementedError` for now.

You usually don't build an index by hand: the store does it for you from the
`index_type` and `index_params` you pass. If you do, an index takes the gallery as
`(paths, vectors)`:

```python
from pyvisim.retrieval import ImageIndexIVFFlat

index = ImageIndexIVFFlat(paths, vectors, quantizer="inner_product", nlist=100, nprobe=8)
```

## Going lower level

If you don't need the `ImageRetriever` façade, call the function directly. It takes
the query images and the store:

```python
from pyvisim.functional import retrieve_top_k_similar

results = retrieve_top_k_similar(query_images, store, k=5)
```
