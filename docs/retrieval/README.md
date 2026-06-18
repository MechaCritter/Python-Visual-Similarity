# pyvisim.retrieval

File: [`pyvisim/retrieval/__init__.py`](../../pyvisim/retrieval/__init__.py)

The `retrieval` package finds the images in a gallery that look most like a query
image. The gallery is an [`ImageEncodingMap`](../image_store.md); the search runs
either by brute-force cosine or through a significantly faster, which is the
recommended way.

## Quick start

Wrap your encoding map in an index, hand it to an `ImageRetriever`, and search:

```python
from pyvisim.retrieval import ImageIndexIVFFlat, ImageRetriever

# encoding_map is an ImageEncodingMap built over your gallery images
index = ImageIndexIVFFlat(encoding_map, quantizer="inner_product", nlist=100, nprobe=8)
retriever = ImageRetriever(index)

results = retriever.retrieve_top_k_similar([query_a, query_b], k=5)
for ranked in results:                 # one list per query image, in input order
    for candidate in ranked:           # already sorted, best match first
        print(candidate.path, candidate.score)
```

You pass the query image array(s), and you get back one ranked `list[Candidate]`
per query. Each `Candidate` is a small named tuple of `(path, score)`.

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

Both expose their coarse-quantizer centroids via `index.cluster_centers`.

## Going lower level

If you don't need the `ImageRetriever` façade, call the function directly. It
takes the gallery and encoder explicitly, and an optional `index`:

```python
from pyvisim.functional import retrieve_top_k_similar

# brute-force cosine over the whole gallery
results = retrieve_top_k_similar(query_images, encoding_map, encoder, k=5)

# or run the search through an index
results = retrieve_top_k_similar(query_images, encoding_map, encoder, k=5, index=index)
```
