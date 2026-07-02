# Siamese Neural Network for Image Similarity

A PyTorch implementation of a **Siamese Neural Network (SNN)** for image similarity. This implementation supports image embedding generation, similarity scoring, and end-to-end training on the Oxford Flowers dataset using a ResNet-18 backbone.


## Overview

Siamese Neural Networks are designed to learn a meaningful embedding space where:

* Similar images are mapped close together.
* Dissimilar images are mapped far apart.

Instead of performing image classification, the model learns feature representations that can be compared using a similarity metric such as cosine similarity. The application of this includes *Image Retrieval*, *One-Shot Learning*, *Face Verification*, *Medical Image Similarity*, *Visual Search* etc.

## Architecture

```text
Input Image A ──► Backbone ──► Embedding Head ──► Embedding A
                      │
                      │ Shared Weights
                      │
Input Image B ──► Backbone ──► Embedding Head ──► Embedding B

Embedding A + Embedding B
            │
            ▼
     Contrastive Loss
```

The siamese neural network implementation consists of:

1. **Backbone Network**:

A feature extraction network (e.g., ResNet-18).

2. **Projection Head**:

A fully connected layer that projects backbone features into a lower-dimensional embedding space.

3. **Embedding Normalization**:

L2 normalization is applied to embeddings.

This architecture further helps to achieve:
* Stable training
* Unit-length feature vectors
* Efficient cosine similarity computation

## References

1. **Siamese Neural Networks for One-shot Image Recognition**
https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

2. **Dimensionality Reduction by Learning an Invariant Mapping** (Hadsell, Chopra, & LeCun, 2006)
http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

3. **Deep Residual Learning for Image Recognition**
https://arxiv.org/abs/1512.03385
