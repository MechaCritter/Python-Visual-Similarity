# Features

Directory: [`pyvisim/features/`](../../pyvisim/features)

A feature extractor maps one image to a `(N, D)` array of local descriptors. Embedders
consume these descriptors and aggregate them into a fixed-size vector:

```text
image -> feature extractor -> local descriptors -> embedder -> embedding
```

The descriptors are what makes the embedding meaningful, so the extractor is the first
thing to tune when retrieval or clustering quality disappoints. Hand-crafted descriptors
(`SIFT`, `RootSIFT`) need no training data, while `DeepConvFeature` reads the feature maps
of a pretrained CNN (ResNet, VGG, EfficientNet, ...) and therefore carries whatever the
backbone learned.

| Object | `output_dim` | Notes |
|--------|--------------|-------|
| [`SIFT`](sift.md) | 128 | SIFT descriptors |
| [`RootSIFT`](rootsift.md) | 128 | SIFT with Hellinger normalization (default extractor) |
| [`DeepConvFeature`](deep_conv_feature.md) | layer channels (+2) | CNN feature maps, optional spatial coordinates |
| [`Lambda`](lambda.md) | user-defined | wraps any custom function |
