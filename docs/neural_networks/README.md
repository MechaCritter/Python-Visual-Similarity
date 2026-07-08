# Neural networks

File: [`neural_networks/`](../../pyvisim/neural_networks/)

Two models live here:

* A **Siamese network** for image similarity. It runs two images through
  a shared backbone (currently only supports `ResNet-18`) and a projection head, L2-normalizes the resulting embeddings, and
  trains them with a contrastive loss so similar images land close together and dissimilar ones
  end up far apart.
* **`ClipEmbedder`**, OpenAI's pretrained CLIP loaded straight from the official
  TorchScript checkpoints (`ViT-B/32`, `ViT-B/16` or `ViT-L/14`). Downloads are
  verified against the published SHA-256 digests and cached, and there's no
  training step: call `embed(images)` and you get L2-normalized embeddings back.

Everything here needs the `nn` extra: `pip install "pyvisim[nn]"`.

For install notes, code examples, and the training walkthrough, see the
[module README](../../pyvisim/neural_networks/README.md).
