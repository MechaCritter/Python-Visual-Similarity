# Encoders

## Overview
The "encoders" module is responsible for computing image similarities, or more precisely the vector representations of visual content in
images which can be used for tasks such as indexing, retrieval, clustering, and classification of images. Most encoders in this module utilize
a combination of feature extraction techniques, clustering models, and a certain similarity function to generate these vector representations,
depending on the specific implementation of the encoder. The CLIP encoder is the exception: it is a pretrained neural network that maps an
image directly to an embedding.
This module currently contains these core components:
- VLAD (Vector of Locally Aggregated Descriptors) Encoder
- Fisher Vector Encoder
- CLIP Encoder
- Similarity Metric Pipeline

ALl the feature extraction classes/methods are implemented in the `features` module.

## VLAD Encoder and Fisher Vector Encoder
The VLAD (Vector of Locally Aggregated Descriptors) and the Fisher Vector Encoders are two similar implementations of image descriptor
encoding designed to extract local image descriptors and compute their aggregated representations, but they
differ in the way they aggregate these descriptors and the underlying clustering methods they use:

- VLAD Encoder: Capture only the first-order statistics of the local features. `KMeans` clustering is used to cluster
  the local features.
  The output has shape (K * D)<sup>[1](#references)</sup>, where K is the number of clusters and D is the
  dimensionality of the local features.
- Fisher Vector Encoder: Capture both first-order and second-order statistics of the local features.
  `Gaussian Mixture Model (GMM)` is used to cluster the local features.
  The output has shape (2 * K * D + K)<sup>[1](#references)</sup> in `scikit-image` implementation (which is also used
  in this project).

After the feature extraction step, the local features are aggregated to their respective cluster centers. The final
encoding matrix is then flattened and normalized to produce the final feature vector representation of the image.

## Configuring Encoders

The encoders build their clustering models internally: VLAD always uses K-Means and the Fisher Vector encoder always
uses a Gaussian Mixture Model (both implemented in `pyvisim.clustering`).

```python
from pyvisim.encoders import VLADEncoder, FisherVectorEncoder

vlad = VLADEncoder(
    n_clusters=256,
    kmeans_params={"random_state": 42},
    pca_params={"n_components": 64},
)
fisher = FisherVectorEncoder(
    n_components=256,
    gmm_params={"random_state": 42},
)
```

Calling `learn(images)` fits the configured PCA (if any) and the clustering model. A fitted encoder can be saved to
disk and restored later:

```python
vlad.learn(images)
path = vlad.save_to_disk("vlad")  # writes vlad.encoder
vlad = VLADEncoder.load_from_disk(path)
```

The `.encoder` file (safetensors) stores everything needed to rebuild the encoder: the fitted clustering model, the
PCA model, the normalization hyperparameters, the similarity metric and the feature-extractor configuration. That's
why `load_from_disk` only needs the path.

pyvisim also ships pretrained encoders. Load one with `VLADEncoder.from_pretrained(PretrainedVLAD.X)` or
`FisherVectorEncoder.from_pretrained(PretrainedFisher.X)`. The older `weights=KMeansWeights.X` / `weights=GMMWeights.X`
constructor argument is deprecated and will be removed in `1.0.0`.

## CLIP Encoder
The `CLIPEncoder` wraps a pretrained [open_clip](https://github.com/mlfoundations/open_clip) model and maps each image
straight to a CLIP embedding, so there is no feature extractor, clustering model, or `learn` step. `open_clip` is a
heavyweight dependency, so it ships as the optional `nn` extra and is imported lazily: importing `pyvisim` never requires
it, and the error (with an install hint) only shows up if you actually build a `CLIPEncoder` without it installed.

```bash
pip install "pyvisim[nn]"
```

```python
from pyvisim.encoders import CLIPEncoder

clip = CLIPEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
embeddings = clip.encode(images)  # (num_images, D); L2-normalized by default
```

The weights download on first use and are cached by open_clip. Saving stores only the model identifiers (not the
weights), so `.encoder` files stay tiny and `load_from_disk` re-fetches the weights to reproduce the original encodings.
See [the CLIP docs](../../docs/encoders/clip.md) for the full rundown.

## Similarity Metric Pipeline
The _Pipeline_ class is designed to handle multiple encoders simultaneously to compute feature vectors. It takes
a list of encoders (instances of the ImageEncoderBase class defined in the '_base_encoder.py' file) and a function
to compute similarity. The pipeline encodes an image using all the encoders included, flatten the resulting
encoding vectors and concatenate them into a single feature vector, which are then fed into the similarity function.

## References
[1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
