# Pipeline

File: [`pipeline.py`](../../pyvisim/encoders/pipeline.py)

`Pipeline` combines several encoders into one. It encodes an image with every member
encoder, concatenates the per-encoder vectors, and compares the combined vectors with
a single similarity function. The goal is a more robust representation that blends,
for example, VLAD and Fisher Vector encodings.

It implements `SimilarityMetric` (not `ImageEncoderBase`), so it exposes `encode`,
`generate_encoding_map`, and `similarity_score` but has no clustering model of its own.

## How `encode` works

1. Validate that every member is an `ImageEncoderBase` (rejected otherwise).
2. Because the input `images` may be a one-shot iterator, it is duplicated with
   `itertools.tee` so each encoder sees the full sequence.
3. Each member's output is temporarily forced to `flatten=True`, encoded, then the
   member's original `flatten` setting is restored. Flattening is mandatory here
   because different encoders produce different output sizes, and concatenation needs
   1D vectors per image.
4. The per-encoder results are concatenated with `np.hstack` into one wide vector per
   image.

## Notes

- Member encoders can use different feature extractors and clustering models; the
  pipeline does not require them to agree, since their outputs are simply concatenated.
- The similarity function is guarded the same way as in the encoders (probed on
  assignment, with a row-wise fallback). Default is cosine similarity.
- A commented-out `fit` method exists in the source; training is done per encoder, not
  through the pipeline.
