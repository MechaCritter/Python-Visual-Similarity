# Pipeline

File: [`pipeline.py`](../../pyvisim/encoders/pipeline.py)

`Pipeline` combines several encoders into one. It encodes an image with every member
encoder, concatenates the per-encoder vectors, and compares the combined vectors with
a single similarity function. The goal is a more robust representation that blends,
for example, VLAD and Fisher Vector encodings.

It implements `SimilarityMetric` (not `ImageEncoderBase`), so it exposes `encode` and
`similarity_score` but has no clustering model of its own. It's also serialisable:
`to_dict`/`from_dict` round-trip the whole pipeline (each member encoder is serialised
in turn), which is what lets an
[`InMemoryImageEmbeddingStore`](../image_store.md) persist a pipeline alongside its
gallery.

## Notes

- Member encoders can use different feature extractors and clustering models; the
  pipeline does not require them to agree, since their outputs are simply concatenated.
- The similarity metric is chosen by name, just like in the encoders: `"cosine"`
  (default), `"euclidean"`, `"l1"` or `"manhattan"`.
- A commented-out `fit` method exists in the source; training is done per encoder, not
  through the pipeline.
- To index a gallery with a pipeline, hand it to an
  [`InMemoryImageEmbeddingStore`](../image_store.md) just like any other encoder.
