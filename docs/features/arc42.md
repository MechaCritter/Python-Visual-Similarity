# arc42: features

Software architecture of `pyvisim.features`. This document is for developers
and is not part of the published documentation.

## Building block view

A feature extractor is one end of the contract the embedders are written
against:

```
image -> feature extractor -> local descriptors -> embedder -> embedding
```

Calling an extractor with a single image returns an `(N, D)` array of local
descriptors. `output_dim` declares `D` ahead of the call, so an embedder can
validate an extractor against its PCA and its clustering model before a single
descriptor is computed. Nothing else is required of an extractor, so `SIFT`,
`RootSIFT`, `DeepConvFeature` and a `Lambda` around an arbitrary function are
interchangeable from an embedder's point of view.

`Lambda` exists because that contract is small enough to satisfy without
subclassing `FeatureExtractorBase`. It is also the reason `output_dim` is a
constructor argument there: an arbitrary function has no inspectable descriptor
size, so the caller has to supply the value.

`FeatureExtractorBase` inherits the file contract of `SerializerMixin`, as the
embedders and the clustering models do. An extractor is described by its
constructor arguments under `"config"`, and `FeatureExtractorBase.from_dict`
rebuilds the class named in that description. This description is also nested
in the state of every classic embedder, so an extractor is saved either inside
an embedder file or in a file of its own.
