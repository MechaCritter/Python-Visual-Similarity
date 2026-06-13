# Features

File: [`_features.py`](../../pyvisim/features/_features.py)

A feature extractor maps one image to a `(N, D)` array of local descriptors. Encoders
consume these descriptors and aggregate them into a fixed-size vector.

| Object | `output_dim` | Notes |
|--------|--------------|-------|
| [`SIFT`](sift.md) | 128 | SIFT descriptors |
| [`RootSIFT`](rootsift.md) | 128 | SIFT with Hellinger normalization (default extractor) |
| [`DeepConvFeature`](deep_conv_feature.md) | layer channels (+2) | CNN feature maps, optional spatial coordinates |
| [`Lambda`](lambda.md) | user-defined | wraps any custom function |
