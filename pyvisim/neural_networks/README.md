# Neural Networks for Image Similarity

This module ships two PyTorch models:

* A **Siamese Neural Network (SNN)**: image embedding generation, similarity scoring, and
  end-to-end training on the Oxford Flowers dataset using a ResNet-18 backbone.
* A **CLIP embedder** (`ClipEmbedder`): pretrained CLIP embeddings from pyvisim's own CLIP
  implementation, with safetensors weights fetched from the Hugging Face Hub. See
  [CLIP Embedder](#clip-embedder) below.


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

## Usage

The Siamese network lives in the `nn` extra, so install that first:

```bash
pip install "pyvisim[nn]"
```

### Scoring image similarity

`SiameseNeuralNetwork` takes NumPy images (HWC, `uint8` in `[0, 255]` by default) and gives back L2-normalized embeddings.

```python
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import SiameseNeuralNetwork

# NumPy HWC uint8 images from the Oxford Flower dataset.
dataset = OxfordFlowerDataset()
image1, *_ = dataset[0]
image2, *_ = dataset[1]

model = SiameseNeuralNetwork(backbone="resnet18", embedding_dim=128, device="cuda")

score = model.similarity_score(image1, image2).item()  # (1, 1) matrix -> Python float
print(f"Similarity: {score:.4f}")
```

Need the raw vectors instead of a score? Use `embed`, which stacks everything into a single
forward pass:

```python
embeddings = model.embed([image1, image2])  # (2, 128) array, one row per image
```

Both methods take a single image, a batched array, or an iterable of images. If your images
aren't HWC `uint8`, tell them the layout and value range with `dims=` and `value_range=` (for
example `dims="CHW"` for PyTorch-style tensors, or `value_range=(0.0, 1.0)` for floats).

### Training on your own data

`ContrastiveLoss` pulls the embeddings of similar pairs together and pushes dissimilar ones
apart. Label `1` marks a matching pair and `0` a non-matching one:

```python
from pyvisim.neural_networks.losses import ContrastiveLoss

criterion = ContrastiveLoss(margin=1.0)
loss = criterion(emb_a, emb_b, labels)  # labels: 1 = similar, 0 = dissimilar
```

Because the embeddings are L2-normalized, the largest possible distance between two of them
is 2, so `margin` has to stay in `(0, 2]`.

There's a ready-to-run script that trains on the Oxford Flowers dataset. It mines
positive/negative pairs, trains with `AdamW` + cosine annealing, and checkpoints the best
model to `checkpoints/best.pt`:

```bash
python -m pyvisim.neural_networks.scripts.train_siamese_neural_network
```

To score two images with a trained checkpoint, reach for the script's `demo` helper:

```python
from pyvisim.neural_networks.scripts.train_siamese_neural_network import demo

demo("checkpoints/best.pt", "flower_a.jpg", "flower_b.jpg")
```

Loading a checkpoint back into a model yourself? Pass `pretrained_backbone=False` so you
don't download the ImageNet weights just to overwrite them:

```python
import torch

model = SiameseNeuralNetwork(embedding_dim=128, pretrained_backbone=False)
model.load_state_dict(torch.load("checkpoints/best.pt")["model"])
```

## CLIP Embedder

`ClipEmbedder` gives you pretrained CLIP embeddings without any training step. The CLIP
image towers (Vision Transformer and modified ResNet) are implemented right here in
pyvisim, and the pretrained weights are downloaded as safetensors files from the
Hugging Face Hub on first use. No third-party CLIP library is involved.

```python
from pyvisim.neural_networks import ClipEmbedder

embedder = ClipEmbedder("ViT-B-32", pretrained="openai")
embeddings = embedder.embed(images)  # (num_images, 512); L2-normalized by default
score = embedder.similarity_score(image1, image2)  # cosine similarity by default
```

A few things worth knowing:

* Variant names and pretrained tags follow open_clip: 67 (variant, tag) combinations
  across 30 variant names are supported, from `RN50`/`ViT-B-32` up to `ViT-g-14` and
  `ViT-bigG-14`, with weights by OpenAI (`"openai"`), LAION (`"laion2b_s34b_b79k"`, ...),
  DataComp (`"datacomp_xl_s13b_b90k"`, ...) and Meta (`"metaclip_fullcc"`, ...). That is
  every open_clip variant whose image tower is a standard CLIP ViT or ResNet and whose
  Hub repository publishes open_clip-format safetensors weights. Enumerate them with
  `pyvisim.neural_networks.clip.available_variants()` and `available_pretrained(variant)`.
  OpenAI-style spellings (`"ViT-B/32"`) are accepted as aliases.
* Only the image tower is loaded — the text tower's weights are skipped entirely.
* Checkpoints land in the standard Hugging Face cache (`~/.cache/huggingface/hub`), the
  same place open_clip's Hub downloads go, so weights you already pulled that way are
  reused. Every download is integrity-checked by huggingface_hub. Pass `cache_dir=...`
  to cache somewhere else.
* Preprocessing matches open_clip per checkpoint: bicubic resize plus center crop (or a
  squashing resize where the checkpoint calls for it) and normalization with the
  checkpoint's channel statistics.
* The model runs in `float32` on both CPU and CUDA, so embeddings are reproducible and
  match what open_clip produces at its default precision. QuickGELU-trained checkpoints
  (like all `"openai"` ones) are automatically built with the QuickGELU activation.

## References

1. **Siamese Neural Networks for One-shot Image Recognition**
https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

2. **Dimensionality Reduction by Learning an Invariant Mapping** (Hadsell, Chopra, & LeCun, 2006)
http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

3. **Deep Residual Learning for Image Recognition**
https://arxiv.org/abs/1512.03385

4. **Learning Transferable Visual Models From Natural Language Supervision** (Radford et al., 2021)
https://arxiv.org/abs/2103.00020
