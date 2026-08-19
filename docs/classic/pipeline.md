# Pipeline

File: [`pipeline.py`](../../pyvisim/classic/pipeline.py)

`Pipeline` glues several embedders into one. It embeds an image with every member,
concatenates the per-member vectors, and compares the combined vectors with a single
similarity function. Stacking representations like this was a common trick in the
classic era: VLAD and Fisher Vector make different mistakes, so a concatenation of the
two is often more robust than either alone.

```python
from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder

vlad = VLADEmbedder(n_clusters=64)
fisher = FisherVectorEmbedder(n_components=64)
for embedder in (vlad, fisher):
    embedder.learn(images)

pipeline = Pipeline([vlad, fisher], similarity_func="cosine")
vectors = pipeline.embed(images)          # (num_images, vlad_dim + fisher_dim)
score = pipeline.similarity_score(image1, image2)
```

Members must be `SerializableImageEmbedder` instances, because the pipeline serialises
each of them in turn. Anything else raises a `ValueError`. The pipeline has no
clustering model of its own, so there's nothing to `learn` at the pipeline level: train
each member first, then compose them.

## Notes

- Members can use different feature extractors and clustering models. The pipeline
  doesn't require them to agree, since their outputs are just concatenated.
- Member vectors are always flattened before concatenation, since members have
  different output shapes. A member's own `flatten` setting is restored afterwards.
- The similarity metric is chosen by name, like everywhere else: `"cosine"` (default),
  `"euclidean"`, `"l1"` or `"manhattan"`.
- `to_dict`/`from_dict` round-trip the whole pipeline, which is what lets an
  [`InMemoryImageEmbeddingStore`](../image_store.md) persist a pipeline alongside its
  gallery. To index a gallery with one, hand it to the store like any other embedder.
