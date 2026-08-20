# Developer Overview

## Package layout

- [Typing](typing.md): Public types.
- [Distance](distance.md): the distance metrics that compare embeddings.
- [Structural](structural/): SSIM and MSSSIM
- [Classic](classic/): Classical embedding methods pre deep learning era.
- [Image store](image_store.md): Image Store for retrieval
- [Features](features/): Image Feature extractors
- [Retrieval](retrieval/): Image Retrieval tools
- [Neural networks](neural_networks/): Siamese Networks, Triplet Networks, CLIP
embedders, ...
- [Dataset](dataset/): `torch` Datasets

## Design decisions worth knowing

- **Serialization uses safetensors `.embedder` format.**:  
  Pickling is explicitly avoided out of safety reasons.
