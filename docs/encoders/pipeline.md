# Pipeline

File: [`pipeline.py`](../../pyvisim/encoders/pipeline.py)

`Pipeline` combines several encoders into one. It encodes an image with every member
encoder, concatenates the per-encoder vectors, and compares the combined vectors with
a single similarity function. The goal is a more robust representation that blends,
for example, VLAD and Fisher Vector encodings.

It implements `SimilarityMetric` (not `ImageEncoderBase`), so it exposes `encode`,
`generate_encoding_map`, and `similarity_score` but has no clustering model of its own.

## Notes

- Member encoders can use different feature extractors and clustering models; the
  pipeline does not require them to agree, since their outputs are simply concatenated.
- The similarity function is guarded the same way as in the encoders (probed on
  assignment, with a row-wise fallback). Default is cosine similarity.
- A commented-out `fit` method exists in the source; training is done per encoder, not
  through the pipeline.
- `generate_encoding_map(image_paths)` returns a lazy
  [`ImageEncodingMap`](../image_store.md), encoding each image with the full pipeline on
  first access. See [base_encoder.md](base_encoder.md).
