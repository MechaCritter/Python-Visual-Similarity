# TripletNeuralNetwork

Learns an embedding space from *triplets*: an anchor, a positive (same class) and a
negative (different class). Training pulls the anchor towards the positive and pushes
it away from the negative by at least a margin. Embeddings come out L2-normalized, so
cosine similarity is a plain dot product.

```python
from pyvisim.neural_networks import TripletNeuralNetwork

model = TripletNeuralNetwork(backbone="resnet18", embedding_dim=128)
embeddings = model.embed(images)                   # (N, 128)
score = model.similarity_score(image1, image2)     # (1, 1) cosine similarity
```

The three classic branches of the architecture are not three networks. They are one
shared-weight pass, so a batch of anchors, positives and negatives goes through the
same backbone and head:

```text
Anchor   ──┐
Positive ──┼──► Backbone ──► Embedding Head ──► L2 Normalize ──► Embeddings
Negative ──┘     (Shared Weights)
```

## Training with online mining

There is no "build a list of triplets" step here. Hand a *labeled* batch to
`TripletLoss` and it mines the triplets from that batch itself, the way FaceNet does.
Every image in the batch takes a turn as the anchor and its partners are picked from
its neighbours:

```python
import torch
from pyvisim.neural_networks.losses import TripletLoss

criterion = TripletLoss(margin=0.2, mining="semi_hard")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for images, labels in loader:          # images: (B, 3, H, W), labels: (B,)
    loss = criterion(model(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Batch composition matters more than usual. A batch that holds only one class has no
negative to mine, and a batch of all-distinct labels has no positive, so both give an
honest loss of 0. Sample a handful of classes with a few images each (FaceNet's "P
classes x K images" recipe) and the mining has something to work with. The zero loss
still carries a gradient, so a degenerate batch never breaks `loss.backward()`.

## Mining strategies

| `mining` | What it picks | Averaged over | Memory |
|---|---|---|---|
| `"semi_hard"` (default) | For every positive pair, the closest negative that is still farther than the positive. Falls back to the anchor's farthest negative when there is none. | all positive pairs | O(B³) |
| `"batch_hard"` | Per anchor, its farthest positive and its closest negative (Hermans et al., 2017). | anchors with both | O(B²) |
| `"batch_all"` | Every valid triplet, but only the ones that violate the margin count towards the mean. Averaging over all of them would let the trivially satisfied majority wash out the signal. | violating triplets | O(B³) |

Two more knobs:

- `margin` (default `0.2`) is how much farther the negative has to be than the positive
  before a triplet stops costing anything.
- `squared` (default `True`) uses squared Euclidean distances, as in FaceNet. Hermans
  et al. report better convergence with plain distances (`squared=False`), usually
  paired with `"batch_hard"`.

`TripletLoss` does not normalize what you give it. That is `forward`'s job, so the
distances it sees are already those of unit vectors when the embeddings come from a
`TripletNeuralNetwork`.

## Saving and loading

The network is a `SerializableImageEmbedder` like the rest of the module:

```python
path = model.save_to_disk("triplet_resnet18")             # -> triplet_resnet18.embedder
model = TripletNeuralNetwork.load_from_disk(path)
```

Loading rebuilds the architecture without downloading the ImageNet weights, since the
saved ones overwrite them anyway. If you built the network with a custom `transform`,
pass it back in, otherwise the reloaded network preprocesses differently and warns you
about it:

```python
model = TripletNeuralNetwork.load_from_disk(path, transform=my_transform)
```

## Which network do I want?

`TripletNeuralNetwork` and [`ContrastiveSiameseNetwork`](contrastive_siamese.md) both
produce an L2-normalized embedding space, and both score images with a fixed metric at
inference time. The difference is what they optimize:

- Contrastive works on *pairs* and pins down absolute distances. Similar pairs are
  pushed to distance 0, dissimilar ones out past the margin.
- Triplet works on *relative* distances. It only asks that the positive be closer than
  the negative, which leaves the space freer to spread classes out at whatever scale
  suits them. That usually holds up better for retrieval over many classes.

Reach for [`BCESiameseNetwork`](bce_siamese.md) instead if you want a learned
same-or-different probability rather than an embedding space.

## References

1. **Deep Metric Learning Using Triplet Network** (Hoffer & Ailon, 2014)
https://arxiv.org/abs/1412.6622

2. **FaceNet: A Unified Embedding for Face Recognition and Clustering** (Schroff,
Kalenichenko, & Philbin, 2015)
https://doi.org/10.1109/CVPR.2015.7298682

3. **In Defense of the Triplet Loss for Person Re-Identification** (Hermans, Beyer, &
Leibe, 2017)
https://arxiv.org/abs/1703.07737
