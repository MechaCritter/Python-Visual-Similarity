# Pretrained weights

File: [`_base_encoder.py`](../../pyvisim/encoders/_base_encoder.py)

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
clustering weight, and `ImageEncoderBase.__init__` loads the PCA automatically when a
`PCA` weight is selected. This is why you never reference `_PCA` directly: choosing a
`*_PCA` clustering weight pulls in the correct PCA for you.

## Adding your own weights

To use a model you trained yourself, skip the enums and pass the fitted model directly
via the encoder's `kmeans_model` / `gmm_model` (and optional `pca`) arguments. See
[base_encoder.md](base_encoder.md).
