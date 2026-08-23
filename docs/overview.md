# Developer Overview

## Package layout

- [Typing](typing.md): Public types.
- [Distance](distance.md): the distance metrics that compare embeddings.
- [Structural](structural/): SSIM and MSSSIM
- [Pixelwise](pixelwise/): PSNR
- [Classic](classic/): Classical embedding methods pre deep learning era.
- [Image store](image_store.md): Image Store for retrieval
- [Features](features/): Image Feature extractors
- [Retrieval](retrieval/): Image Retrieval tools
- [Neural networks](neural_networks/): Siamese Networks, Triplet Networks, CLIP
embedders, ...
- [Dataset](dataset/): `torch` Datasets

The abstract bases every public class derives from live in two places:
`pyvisim/_base_classes.py` (`SimilarityMetric`, `FeatureExtractorBase`,
`ImageEmbedderBase`, `SerializableImageEmbedder`) and `pyvisim/base/`
(`DenseMetricBase`, shared by the dense metrics).

## Design decisions worth knowing

- **Serialization uses safetensors `.embedder` format.**:  
  Pickling is explicitly avoided out of safety reasons.

## TODO

- Move `pyvisim/_base_classes.py` into `pyvisim/base/`, so that all abstract
  bases live in one module.
