# Neural networks

File: [`neural_networks/`](../../pyvisim/neural_networks/)

This module implements **Siamese networks** for image similarity. Both variants run two images
through a shared backbone (currently only supports `ResNet-18`) and a projection head, and then
differ in how the branches are compared:

- `ContrastiveSiameseNetwork` L2-normalizes the resulting embeddings and trains them with a
  contrastive loss so similar images land close together and dissimilar ones end up far apart;
  similarity is a fixed metric (cosine by default) on the embeddings.
- `PairwiseSiameseNetwork` feeds the component-wise L1 distance of sigmoid-activated branch
  features into a learned scoring layer and returns the probability that both images show the
  same class (Koch et al., 2015); it trains as a binary classifier over pairs.

`SiameseNeuralNetwork` remains as a deprecated alias of `ContrastiveSiameseNetwork`.

Everything here needs the `nn` extra: `pip install "pyvisim[nn]"`.

For install notes, code examples, and the training walkthrough, see the
[module README](../../pyvisim/neural_networks/README.md).
