# Pretrained encoders

File: [`_base_encoder.py`](../../pyvisim/encoders/_base_encoder.py)

pyvisim bundles pretrained encoders trained on the Oxford Flowers dataset with
`k=256`. Load one with `from_pretrained` and the matching enum:

```python
from pyvisim.encoders import VLADEncoder, FisherVectorEncoder, PretrainedVLAD, PretrainedFisher

vlad = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)
fisher = FisherVectorEncoder.from_pretrained(PretrainedFisher.OXFORD102_K256_VGG16_PCA)
```

Each enum member points to a bundled `.encoder` file under
`pyvisim/res/model_files/`. `from_pretrained` just calls `load_from_disk` on
that path, so you get a fully rebuilt encoder, feature extractor and similarity
metric included, with nothing else to pass in. Loading a file saved by a
different encoder class raises, so `VLADEncoder.from_pretrained` won't accept a
`PretrainedFisher` member, and vice versa.

## Available enums

- `PretrainedVLAD`: pretrained VLAD encoders (defined in `vlad.py`).
- `PretrainedFisher`: pretrained Fisher Vector encoders (defined in `fisher_vector.py`).

Each enum has the same six variants, covering three feature types (VGG16 deep
features, RootSIFT, SIFT) each with and without PCA, for example
`OXFORD102_K256_ROOTSIFT_PCA`. Variants ending in `_PCA` reduce the descriptor
dimensions by half with PCA before clustering.

## Deprecated: the `weights=` enums

> ⚠️ **Deprecated.** Passing `weights=KMeansWeights.X` / `weights=GMMWeights.X`
> to an encoder constructor emits a `DeprecationWarning` and will be removed in
> `1.0.0`. Use `from_pretrained()` (or `load_from_disk()`) instead. Everything
> tagged `# TODO: removed with version 1.0.0` goes away then.

`KMeansWeights`, `GMMWeights` and the internal `_PCA` enum still exist for the
legacy path. They now point at the same bundled `.encoder` files, and
`_load_pretrained_weights` rebuilds the clustering model (and PCA, via
`_CLUSTERING_TO_PCA_MAPPING`) from them with `load_encoder_state` +
`ClusteringModel.from_dict`. They load the exact same trained models as the
`Pretrained*` enums, so migrating is just a name swap.

## Using your own trained model

Configure the encoder from parameters, call `learn()` on your images, and save
the result with `save_to_disk()`; reload it later with `load_from_disk()`. The
`.encoder` file round-trips the clustering model, PCA, normalization settings,
similarity metric and feature extractor. See [base_encoder.md](base_encoder.md).
