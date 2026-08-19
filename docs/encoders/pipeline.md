# Pipeline

File: [`pipeline.py`](../../pyvisim/encoders/pipeline.py)

`Pipeline` combines several embedders into one. It embeds an image with every member
embedder, concatenates the per-embedder vectors, and compares the combined vectors with
a single similarity function. The goal is a more robust representation that blends,
for example, VLAD and Fisher Vector embeddings.

It implements `SimilarityMetric` (not `ImageEmbedderBase`), so it exposes `embed` and
`similarity_score` but has no clustering model of its own. It's also serialisable:
`to_dict`/`from_dict` round-trip the whole pipeline (each member embedder is serialised
in turn), which is what lets an
[`InMemoryImageEmbeddingStore`](../image_store.md) persist a pipeline alongside its
gallery.

## Notes

- Member embedders can use different feature extractors and clustering models; the
  pipeline does not require them to agree, since their outputs are simply concatenated.
- The similarity metric is chosen by name, just like in the embedders: `"cosine"`
  (default), `"euclidean"`, `"l1"` or `"manhattan"`.
- A commented-out `fit` method exists in the source; training is done per embedder, not
  through the pipeline.
- To index a gallery with a pipeline, hand it to an
  [`InMemoryImageEmbeddingStore`](../image_store.md) just like any other embedder.
