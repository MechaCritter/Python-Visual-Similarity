# Neural networks

File: [`neural_networks/`](../../pyvisim/neural_networks/)

This module implements a **Siamese network** for image similarity. It runs two images through
a shared backbone (currently only supports `ResNet-18`) and a projection head, L2-normalizes the resulting embeddings, and
trains them with a contrastive loss so similar images land close together and dissimilar ones
end up far apart.

Everything here needs the `nn` extra: `pip install "pyvisim[nn]"`.

For install notes, code examples, and the training walkthrough, see the
[module README](../../pyvisim/neural_networks/README.md).
