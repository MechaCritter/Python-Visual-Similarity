# Neural networks

File: [`neural_networks/`](../../pyvisim/neural_networks/)

Two models live here:

* A **Siamese network** for image similarity. It runs two images through
  a shared backbone (currently only supports `ResNet-18`) and a projection head, L2-normalizes the resulting embeddings, and
  trains them with a contrastive loss so similar images land close together and dissimilar ones
  end up far apart.
* **`ClipEmbedder`**, pretrained CLIP embeddings from pyvisim's own implementation of
  the CLIP image towers. Weights come as safetensors files from the Hugging Face Hub,
  with open_clip-style variant names and pretrained tags (`"ViT-B-32"` + `"openai"`,
  `"laion2b_s34b_b79k"`, ...; 67 combinations supported). There's no training step:
  call `embed(images)` and you get L2-normalized embeddings back.

Everything here needs the `nn` extra: `pip install "pyvisim[nn]"`.

For install notes, code examples, and the training walkthrough, see the
[module README](../../pyvisim/neural_networks/README.md).
