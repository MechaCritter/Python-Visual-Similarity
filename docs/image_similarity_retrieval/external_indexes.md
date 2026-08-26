# Using index from external libraries

Libraries such as [FAISS](https://github.com/facebookresearch/faiss) implement various other indexing algorithms that offer
much more flexibility than the baseline of `pyvisim`. To allow
such indexes to be used for image similarity search in this library, the `ExternalSearchIndex` class is provided.

> [!NOTE]
> Like the case of `hnsw` index, each index created can potentially increase memory usage, since each index has
its own internal structure, built from the embeddings themselves (which, in the worst case, can be the **same as the original embeddings**).

An example use-case is provided below:

```python
import faiss
from pyvisim.image_store import ExternalSearchIndex, InMemoryImageEmbeddingStore

faiss_index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(vectors)
faiss_index.add(vectors)

store = InMemoryImageEmbeddingStore(
    gallery_paths,              # one path per indexed vector, same order
    embedder,                   # still needed, to embed the queries
    ExternalSearchIndex.from_faiss_index(faiss_index, name="flat-ip"),
)
```

> [!IMPORTANT]
> - **Normalization must be done by the user.** An index built for `METRIC_INNER_PRODUCT` ranks by
  cosine similarity only if vectors are normalised before being added, and the
  embeddings of query images must be normalised the same way.
> - **The scores stay the index's own.** An L2 index returns distances (lower is better),
an inner-product index returns similarities (higher is better).
> - **Since some external indexes are lossy, reconstruction is not always possible.** In such cases, the original vectors must be passed explicitly to `save_to_disk`. The index
itself cannot be written to disk either, so a rebuilt one is passed back to `load_from_disk`:
> ```python
> store.save_to_disk("gallery.safetensors", vectors=original_embeddings)
> restored = InMemoryImageEmbeddingStore.load_from_disk(
>     "gallery.safetensors",
>     search_index=ExternalSearchIndex.from_faiss_index(faiss_index, name="flat-ip"),
> )
> ```
