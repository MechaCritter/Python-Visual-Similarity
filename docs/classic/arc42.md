# arc42: classic

Software architecture of `pyvisim.classic`. This document is for developers and
is not part of the published documentation.

## Building block view

An embedder in this module aggregates local descriptors and is built from three
replaceable parts:

- a feature extractor that produces the local descriptors (`RootSIFT` by
  default),
- an optional PCA, fitted on the descriptors before the clustering model sees
  them,
- a clustering model whose fitted parameters are the vocabulary the embedding
  is computed against (`KMeans` for `VLADEmbedder`, a Gaussian Mixture Model
  for `FisherVectorEmbedder`).

`learn` fits these in order: the PCA first, if there is one, then the
clustering model on the projected descriptors. `Pipeline` composes several
fitted embedders by concatenating their vectors, so it owns no vocabulary of
its own.

The clustering models are internal (`pyvisim.classic._clustering`), so their
parameters reach a caller only through the `pca_params`, `kmeans_params` and
`gmm_params` dictionaries that the embedders forward.
