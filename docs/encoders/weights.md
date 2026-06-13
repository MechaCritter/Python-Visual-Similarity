# Pretrained weights

File: [`_base_encoder.py`](../../pyvisim/encoders/_base_encoder.py)

> ⚠️ **Deprecated.** Loading pretrained models through the `KMeansWeights` / `GMMWeights`
> enums now emits a `DeprecationWarning` and will be removed in a future release. Train
> an encoder with `learn()` and persist it with `save_to_disk()` / `load_from_disk()`
> instead, see [base_encoder.md](base_encoder.md). The rest of this page documents the
> legacy path while it's still around.

Pretrained clustering models are exposed as enums so users can select a model by name
instead of handling file paths. Each enum member's value is a path to a pickled
scikit-learn model under `pyvisim/res/model_files/`, and the shared `_PretrainedModels`
base provides `.load()` to unpickle it with `joblib`.

## Available enums

- `KMeansWeights`: pretrained `KMeans` models for `VLADEncoder`.
- `GMMWeights`: pretrained `GaussianMixture` models for `FisherVectorEncoder`.
- `_PCA`: internal; PCA models. Not exported.

All models were trained on the Oxford Flowers dataset with `k=256`. Each enum has the
same six variants, covering three feature types (VGG16 deep features, RootSIFT, SIFT)
each with and without PCA, for example `OXFORD102_K256_ROOTSIFT_PCA`.

## Automatic PCA pairing

A weight name containing `PCA` requires the matching PCA model so descriptors are
reduced before clustering. `_CLUSTERING_TO_PCA_MAPPING` maps each `_PCA` variant to its
clustering weight, and `_load_pretrained_weights` loads the PCA automatically when a
`PCA` weight is selected. This is why you never reference `_PCA` directly: choosing a
`*_PCA` clustering weight pulls in the correct PCA for you.

The pickled scikit-learn estimators aren't used raw. Each one is adopted into the
matching [`pyvisim.clustering`](../clustering/README.md) model via its internal
`_from_sklearn` classmethod, which type-checks the estimator (and re-validates the
diagonal covariance for the GMM) before wrapping it.

## Using your own trained model

Don't reach for the enums for this. Configure the encoder from parameters, call
`learn()` on your images, and save the result with `save_to_disk()`; reload it later
with `load_from_disk()`. That's the supported replacement for the weight enums, and it
round-trips the PCA and normalization settings too. See [base_encoder.md](base_encoder.md).
