<!-- Logo -->
<p align="center">
  <img src="res/images/logo.png" alt="pyvisim" width="1418" />
</p>

<!-- Added badges to convey project readiness/branding (example placeholders) -->
![License](https://img.shields.io/github/license/MechaCritter/Python-Visual-Similarity)
![Version](https://img.shields.io/pypi/v/pyvisim)
![Status](https://img.shields.io/badge/status-pre--release-orange)
![Python](https://img.shields.io/pypi/pyversions/pyvisim)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

# Welcome to `pyvisim`!

`pyvisim` is a Python library for computing image similarities using image embedders
and neural networks.

📚 **Documentation**: <https://mechacritter.github.io/Python-Visual-Similarity/>

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
4. [Contributing](#contributing)
5. [Get in Touch](#get-in-touch)
6. [TODO](#todo)
7. [License](#license)
8. [References](#references)


## Status

> [!WARNING]
> This project is still in early development, so the API might change anytime (with deprecation,
> but the change will come soon afterwards). Feel free to use it in development environments, but I
> would recommend against using it in production.
>
> The first stable release will have the version tag `v1.0.0` and will come approximately by the
> end of `August 2026`.

## Overview

TODO: add diagram showing how image embedding works

The goal of `pyvisim` is to become the largest collection of image similarity metrics in Python, varying from
traditional methods like `PSNR`, `SSIM`, `Fisher Vectors`, and `VLAD` to deep learning methods like `CLIP` and `Siamese Networks`. Then, one can use these for image retrieval and clustering.

Currently, one would need to install numerous libraries just to get all the metrics mentioned (for example, `scikit-image` + `opencv-python` for `Fisher Vectors` and `SSIM`, `open-clip` for `CLIP Embedder`). `pyvisim`
depends on none of those. All the metrics are implemented using only `numpy`, `scipy` (for conventional metrics), and
`torch` (for deep learning metrics).

### Accelerated Computation

**Cython** kernels are used for some metrics to accelerate computation significantly compared
to all reference libraries on the CPU. See, for example, [benchmark results of the `SSIM` implementation](docs/structural/README.md#benchmarking).

### Examples

#### Structural Similarity:

```python
from pyvisim.structural import SSIM

ssim = SSIM()
similarity_score = ssim.similarity_score(image1, image2)
print(f"Similarity Score: {similarity_score}")
```

#### One-Shot similarity computation using the `CLIPEmbedder`:

```python
from pyvisim.neural_networks import ClipEmbedder

# Declare the Clip Embedder
embedder = ClipEmbedder()

# Compute the similarity score. By default, cosine similarity is used.
similarity_score = embedder.similarity_score(image1, image2)
print(f"Similarity Score: {similarity_score}")
```

#### Image retrieval:

```python
from pyvisim.neural_networks import ClipEmbedder
from pyvisim.image_store import InMemoryImageEmbeddingStore

embedder = ClipEmbedder()

image_store = InMemoryImageEmbeddingStore(
    image_paths=train_image_paths,
    embedder=embedder,
    search_index="hnsw",
    index_params={"m": 16, "ef_construction": 200},
)

candidates = image_store.retrieve_top_k_similar(image, k=5)[0] # returns a tuple of (image_path, similarity_score)
```

For more examples, please refer to the [`pyvisim` Examples
Repository](https://github.com/MechaCritter/Python-Visual-Similarity-Examples).

## Installation

To install the slim version without heavy deep learning stuff, run:

```bash
pip install pyvisim
```

Additional features include (note: these pull in heavy dependencies like `torch`):

```bash
# For deep learning features and the OxfordFlowerDataset
pip install "pyvisim[nn]"
```

All experiments in this project was made on the Oxford Flower Dataset
<ref>[7]</ref>, for which I have created a custom dataset class. For
more details on the dataset, please refer to the [documentation](pyvisim/datasets/README.md).

## Contributing

We love contributions of all kinds—whether it’s suggesting new features, fixing bugs, or writing docs! Here’s how you
can get involved:

1. **Fork** this repository.  
2. **Create a new branch** for your changes.  
3. **Open a pull request** with a clear description of your idea or fix.

We welcome all feedback and hope to build a supportive community around pyvisim!

## Get in Touch
If you have any questions or just want to say hi, feel free to:
- Open an issue on [GitHub](https://github.com/MechaCritter/similarity_metrics_of_images/issues).
- Write me an email at [vunhathuy234@gmail.com](mailto:vunhathuy234@gmail.com).
- Connect on [LinkedIn](https://www.linkedin.com/in/nhat-huy-vu-80495111b/) to follow my work and share your thoughts.

## TODO

The features below are planned for future releases:

- Add **tensor sketch approximation** and **mutual information** analysis for Fisher Vector, according to this
paper by Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng <sup>[1](#references)</sup>
- Add support for **vision transformers** for the `DeepConvFeature` class.

You are welcome to implement any of these features or suggest new ones!

## License
This project is licensed under the terms of the MIT license.
