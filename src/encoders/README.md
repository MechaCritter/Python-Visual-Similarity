# Encoders 

## Overview
The "encoders" module is responsible for computing image similarities, or more precisely the vector representations of visual content in 
images which can be used for tasks such as indexing, retrieval, clustering, and classification of images. The encoders in this module utilize 
a combination of feature extraction techniques, clustering models, and a certain similarity function to generate these vector representations, 
depending on the specific implementation of the encoder.
This module currently contains three core components:
- VLAD (Vector of Locally Aggregated Descriptors) Encoder
- Fisher Vector Encoder
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

## Similarity Metric Pipeline
The _Pipeline_ class is designed to handle multiple encoders simultaneously to compute feature vectors. It takes 
a list of encoders (instances of the ImageEncoderBase class defined in the '_base_encoder.py' file) and a function
to compute similarity. The pipeline encodes an image using all the encoders included, flatten the resulting 
encoding vectors and concatenate them into a single feature vector, which are then fed into the similarity function.

## References
[1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.