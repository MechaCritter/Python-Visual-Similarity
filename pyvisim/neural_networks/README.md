# Siamese Neural Networks for Image Similarity

PyTorch implementations of **Siamese Neural Networks** for image similarity. Both variants share
the same twin architecture — two images pass through the *same* shared-weight backbone
(e.g. ResNet-18) and projection head — but differ in how the two branch outputs are compared:

* `ContrastiveSiameseNetwork` compares L2-normalized embeddings with a **fixed metric**
  (cosine by default) and is trained with a contrastive loss
  (Hadsell, Chopra & LeCun, 2006).
* `PairwiseSiameseNetwork` **learns the comparison itself**: a scoring layer on top of the
  component-wise L1 distance outputs the probability that the two images show the same class
  (Koch, Zemel & Salakhutdinov, 2015).

Both support image embedding generation, similarity scoring, and end-to-end training on the
Oxford Flowers dataset.

## Overview

Siamese Neural Networks learn from *pairs* of images labelled `1` (matching) or `0`
(non-matching) instead of per-image class labels. The applications include *Image Retrieval*,
*One-Shot Learning*, *Face Verification*, *Medical Image Similarity*, *Visual Search* etc.

Pick the variant by what you need at inference time:

* A **geometric embedding space** you can index, cluster, or search with any vector database →
  `ContrastiveSiameseNetwork`. Similar images are mapped close together, dissimilar ones far
  apart, and similarity is a fixed metric on the embeddings.
* A **calibrated "same class?" probability** for pairs, e.g. verification or one-shot
  recognition → `PairwiseSiameseNetwork`. The network itself scores the pair.

## Architecture

### `ContrastiveSiameseNetwork` (Hadsell, Chopra & LeCun, 2006)

```text
Input Image A ──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embedding A
                      │
                      │ Shared Weights
                      │
Input Image B ──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embedding B

Embedding A + Embedding B
            │
            ▼
     Contrastive Loss (training) / fixed metric, e.g. cosine (inference)
```

L2 normalization yields unit-length feature vectors, stable training, and efficient cosine
similarity (a plain dot product).

### `PairwiseSiameseNetwork` (Koch et al., 2015)

```text
Input Image A ──► Backbone ──► Embedding Head ──► Sigmoid ──► Features A ─┐
                      │                                                   ├─► |A - B| ──► Scoring Layer ──► P(same class)
Input Image B ──► Backbone ──► Embedding Head ──► Sigmoid ──► Features B ─┘
        (Shared Weights)
```

Each branch produces a sigmoid-activated feature vector `h ∈ (0, 1)^D`. The pair is combined by
the component-wise L1 distance and scored by a single learned linear layer:

```text
p(A, B) = sigmoid( Σ_j α_j · |h_A,j - h_B,j| + b )
```

The weights `α_j` learn how much each feature dimension matters for the comparison, so the
metric itself is trained. The score is symmetric, lives in `(0, 1)`, and for two identical
images equals `sigmoid(b)` — the learned bias sets the operating point, so a perfect match
does not score exactly `1`.

## Usage

The Siamese networks live in the `nn` extra, so install that first:

```bash
pip install "pyvisim[nn]"
```

### Scoring image similarity

Both networks take NumPy images (HWC, `uint8` in `[0, 255]` by default).

```python
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import ContrastiveSiameseNetwork, PairwiseSiameseNetwork

# NumPy HWC uint8 images from the Oxford Flower dataset.
dataset = OxfordFlowerDataset()
image1, *_ = dataset[0]
image2, *_ = dataset[1]

model = ContrastiveSiameseNetwork(backbone="resnet18", embedding_dim=128, device="cuda")
score = model.similarity_score(image1, image2).item()  # cosine similarity in [-1, 1]
print(f"Similarity: {score:.4f}")

classifier = PairwiseSiameseNetwork(backbone="resnet18", embedding_dim=128, device="cuda")
probability = classifier.similarity_score(image1, image2).item()  # P(same class) in (0, 1)
print(f"Same-class probability: {probability:.4f}")
```

Need the raw vectors instead of a score? Use `embed`, which stacks everything into a single
forward pass:

```python
embeddings = model.embed([image1, image2])  # (2, 128) array, one row per image
```

For `ContrastiveSiameseNetwork` the rows are L2-normalized embeddings; for
`PairwiseSiameseNetwork` they are the sigmoid-activated branch features in `(0, 1)`.

Both methods take a single image, a batched array, or an iterable of images. If your images
aren't HWC `uint8`, tell them the layout and value range with `dims=` and `value_range=` (for
example `dims="CHW"` for PyTorch-style tensors, or `value_range=(0.0, 1.0)` for floats).

> **Note:** `SiameseNeuralNetwork` is a deprecated alias of `ContrastiveSiameseNetwork` and
> will be removed in a future release; it emits a `FutureWarning` on construction.

### Training on your own data

**Contrastive variant.** `ContrastiveLoss` pulls the embeddings of similar pairs together and
pushes dissimilar ones apart. Label `1` marks a matching pair and `0` a non-matching one:

```python
from pyvisim.neural_networks.losses import ContrastiveLoss

criterion = ContrastiveLoss(margin=1.0)
emb_a, emb_b = model(img_a), model(img_b)  # one forward pass per branch
loss = criterion(emb_a, emb_b, labels)  # labels: 1 = similar, 0 = dissimilar
```

Because the embeddings are L2-normalized, the largest possible distance between two of them
is 2, so `margin` has to stay in `(0, 2]`.

**Pairwise variant.** The network is a binary classifier over pairs: `forward` takes both
image batches at once and returns one *logit* per pair, which composes with PyTorch's
numerically stable `BCEWithLogitsLoss` (the paper's regularized cross-entropy; use the
optimizer's `weight_decay` for the regularization term):

```python
import torch

criterion = torch.nn.BCEWithLogitsLoss()
logits = classifier(img_a, img_b)  # one logit per pair
loss = criterion(logits, labels)  # labels: 1 = same class, 0 = different
```

There are ready-to-run scripts that train each variant on the Oxford Flowers dataset. Both
mine positive/negative pairs the same way and train with `AdamW` + cosine annealing; the
contrastive script checkpoints the best model to `checkpoints/best.pt`, the pairwise one to
`checkpoints_pairwise/best.pt`:

```bash
# Contrastive variant (Hadsell et al., 2006)
python -m pyvisim.neural_networks.scripts.train_siamese_neural_network

# Pairwise variant (Koch et al., 2015)
python -m pyvisim.neural_networks.scripts.train_pairwise_siamese_network
```

To score two images with a trained checkpoint, reach for the scripts' `demo` helpers:

```python
from pyvisim.neural_networks.scripts import (
    train_pairwise_siamese_network,
    train_siamese_neural_network,
)

# Prints the cosine similarity in [-1, 1].
train_siamese_neural_network.demo("checkpoints/best.pt", "flower_a.jpg", "flower_b.jpg")

# Prints the same-class probability in (0, 1).
train_pairwise_siamese_network.demo(
    "checkpoints_pairwise/best.pt", "flower_a.jpg", "flower_b.jpg"
)
```

Loading a checkpoint back into a model yourself? Pass `pretrained_backbone=False` so you
don't download the ImageNet weights just to overwrite them:

```python
import torch

model = ContrastiveSiameseNetwork(embedding_dim=128, pretrained_backbone=False)
model.load_state_dict(torch.load("checkpoints/best.pt")["model"])
```

## References

1. **Siamese Neural Networks for One-shot Image Recognition** (Koch, Zemel, & Salakhutdinov, 2015)
https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

2. **Dimensionality Reduction by Learning an Invariant Mapping** (Hadsell, Chopra, & LeCun, 2006)
http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

3. **Deep Residual Learning for Image Recognition**
https://arxiv.org/abs/1512.03385
