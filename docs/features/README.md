# Features

A feature extractor maps one image to a `(N, D)` array of local descriptors. Embedders
consume these descriptors and aggregate them into a fixed-size vector:

```text
image -> feature extractor -> local descriptors -> embedder -> embedding
```

The table below includes feature extractors currently implemented in `pyvisim`.

| Object | `output_dim` | Notes |
|--------|--------------|-------|
| [`SIFT`](sift.md) | 128 | SIFT descriptors |
| [`RootSIFT`](rootsift.md) | 128 | SIFT with Hellinger normalization (default extractor) |
| [`DeepConvFeature`](deep_conv_feature.md) | layer channels | CNN feature maps |
| [`Lambda`](lambda.md) | user-defined | wraps any custom function |
