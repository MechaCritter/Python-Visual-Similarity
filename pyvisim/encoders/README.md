# Embedders

## Overview
This module implements the VLAD and Fisher Vector, which were popular
image embedding methods before image embeddings became popular. These
metrics aggregate image descriptors (commonly SIFT) using a clustering
algorithm to produce a single feature vector representation of the image.

## VLAD Embedder and Fisher Vector Embedder
VLAD (Vector of Locally Aggregated Descriptors) and Fisher Vector differ in the
way they aggregate extracted descriptors descriptors and the underlying clustering methods they use:

- VLAD Embedder: Capture only the first-order statistics of the local features. `KMeans` clustering is used to cluster
  the local features.
  The output has shape (K * D)<sup>[1](#references)</sup>, where K is the number of clusters and D is the
  dimensionality of the local features.
- Fisher Vector Embedder: Capture both first-order and second-order statistics of the local features.
  A `Gaussian Mixture Model` is used to cluster the local features.
  The output has shape (2 * K * D + K)<sup>[1](#references)</sup>.
- For both, `PCA` can be applied at the feature extraction step to reduce the size
of the final vector. If `PCA` is applied, replace the D in the output shape with the
number of components.

After the feature extraction step, the local features are aggregated to their
respective cluster centers. The final embedding matrix is then flattened and
normalized to produce the final feature vector representation of the image.

## Configuring Embedders

The embedders build their clustering models internally: VLAD always uses K-Means and the Fisher Vector embedder always
uses a Gaussian Mixture Model.

```python
from pyvisim.encoders import VLADEmbedder, FisherVectorEmbedder

vlad = VLADEmbedder(
    n_clusters=256,
    kmeans_params={"random_state": 42},
    pca_params={"n_components": 64},
)
fisher = FisherVectorEmbedder(
    n_components=256,
    gmm_params={"random_state": 42},
)
```

Calling `learn(images)` fits the configured PCA (if any) and the clustering model. A fitted embedder can be saved to
disk and restored later:

```python
vlad.learn(images)
path = vlad.save_to_disk("vlad")  # writes vlad.embedder
vlad = VLADEmbedder.load_from_disk(path)
```

## Similarity Metric Pipeline
The _Pipeline_ class is designed to handle multiple embedders simultaneously to compute feature vectors. It takes
a list of embedders (instances of the ImageEmbedderBase class defined in the '_base_embedder.py' file) and a function
to compute similarity. The pipeline embeds an image using all the embedders included, flatten the resulting
embedding vectors and concatenate them into a single feature vector, which are then fed into the similarity function.

## References
[1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
