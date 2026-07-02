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

`pyvisim` is a Python library for computing image similarities using the encoders Fisher Vectors, VLAD
and the Siamese Neural Networks.

## Table of Contents

1. [Installation](#installation)
2. [Why **pyvisim**](#why-pyvisim)
3. [Pretrained Models](#pretrained-models)
4. [Contributing](#contributing)
5. [Get in Touch](#get-in-touch)
6. [TODO](#todo)
7. [License](#license)
8. [References](#references)

For a technical deep-dive into the library internals, see the [developer documentation](docs/overview.md).

## Status

> [!WARNING]
> This project is still in early development, so the API might change anytime (with deprecation,
> but the change will come soon afterwards). Feel free to use it in development environments, but I
> would recommend against using it in production.
>
> The first stable release will have the version tag `v1.0.0` and will come approximately by the
> end of `August 2026`.

## Installation

To use the library, you can simply install it via pip:

```bash
```bash
pip install pyvisim
# For deep learning features and the OxfordFlowerDataset
pip install "pyvisim[nn]"
# For image search feature
pip install "pyvisim[search]"
```
```

or clone the repository and install it locally:

```bash
git clone https://github.com/MechaCritter/Python-Visual-Similarity.git
cd Python-Visual-Similarity
pip install .
```
Note that the *notebooks are only available if you clone the repository.*

All experiments in this project was made on the Oxford Flower Dataset <ref>[7]</ref>, for which I
have created a custom dataset class. To use this class, import it as follows:

```python
from pyvisim.datasets import OxfordFlowerDataset
```
For more details on the dataset, please refer to the [documentation](pyvisim/datasets/README.md).

## Why `pyvisim`?

`pyvisim` is designed to provide a simple and efficient way to compare images.

### Quick Start

With just a few lines of code, you can compute the similarity score between two images using the VLAD encoder:

#### Example: Compute Similarity Score Using VLAD

```python
from pyvisim.encoders import VLADEncoder, PretrainedVLAD
from pyvisim.datasets import OxfordFlowerDataset  # needs "nn" extra: install with `pip install "pyvisim[nn]"`

# Load images from the Oxford Flower Dataset. Has to be NumPy Images!
dataset = OxfordFlowerDataset()
image1, *_  = dataset[0]
image2, *_ = dataset[1]

# Load a bundled pretrained VLAD encoder (RootSIFT features, k=256).
# The feature extractor and similarity metric come with it.
encoder = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)

# Compute the similarity score. By default, cosine similarity is used.
similarity_score = encoder.similarity_score(image1, image2)

print(f"Similarity Score: {similarity_score}")
```

By default the encoder uses cosine similarity. To use a different metric, pass
its name; `"cosine"`, `"euclidean"`, `"l1"` and `"manhattan"` are supported:

```python
encoder = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)
encoder.similarity_func = "euclidean"
```

A fitted encoder can be saved to a `.encoder` file and restored later:

```python
path = encoder.save_to_disk("vlad_oxford102")  # writes vlad_oxford102.encoder
encoder = VLADEncoder.load_from_disk(path)
```
You can also visit the [introduction notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/getting_started.ipynb) for more examples.

I also provided various notebooks for different use-cases. Feel free to check them out, and let me know if you
have any suggestions or questions!

1. **Image Retrieval**  
   Retrieve the top-k most similar images from a dataset.  
   - Use encoding methods like VLAD or Fisher Vectors to quickly find the most relevant matches. Please visit
   [this juptyer notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/vlad_and_fisher_with_vgg16_deep_features.ipynb) for an example.
   - For large galleries, build an `InMemoryImageEmbeddingStore` over your image paths;
     it indexes the embeddings and searches them for you (needs the `search` extra:
     `pip install "pyvisim[search]"`):

     ```python
     from pyvisim.image_store import InMemoryImageEmbeddingStore

     store = InMemoryImageEmbeddingStore(
         gallery_paths, encoder, "ivf-flat",
         quantizer="inner_product", index_params={"nlist": 100},
     )
     results = store.retrieve_top_k_similar(query_images, k=5)
     ```
     See the [retrieval docs](docs/retrieval/README.md) for more information.
   - Example use: Building a fast image search engine for photo management software.

2. **Deep Learning Embeddings**  
   - Generate VLAD or Fisher vectors from neural network embeddings, e.g., VGG16 or other models.
   - Enhance your deep learning pipeline by leveraging traditional encoding methods on top of CNN features.
   - Or skip the aggregation entirely and use `CLIPEncoder` for ready-made CLIP embeddings.
   - The VGG16 deep-feature path (`DeepConvFeature`) and `CLIPEncoder` both need the `nn`
   extra: `pip install "pyvisim[nn]"`.

3. **Image Clustering**  
   - Cluster images based on their similarities to group them by category or content. An example and benchmarking
    can be found in [this notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/clustering_images_using_fv.ipynb).
   - Useful for organizing unlabeled data or generating pseudo-labels for further training.

4. **Pipeline for Combining Multiple Encoders**  
   - Chain various encoders in a single pipeline. An example can be found in [this notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/pipeline.ipynb).
   - Achieve more robust similarity metrics by blending different feature representations.

5. **Siamese Network (Coming Soon!)**  
   - Train a neural network to learn a similarity function directly from pairs/triples of images.  
   - Possible use cases include face recognition, signature verification, or any image-based identity matching.

## Pretrained Models

pyvisim ships ready-to-use pretrained encoders trained on the Oxford-102 flower
dataset. Each one is a bundled `.encoder` file that already includes the right
feature extractor and the cosine similarity metric, so loading one gives you a
working encoder in a single line:

```python
from pyvisim.encoders import (
    VLADEncoder,
    FisherVectorEncoder,
    PretrainedVLAD,
    PretrainedFisher,
)

vlad = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)
fisher = FisherVectorEncoder.from_pretrained(PretrainedFisher.OXFORD102_K256_VGG16_PCA)
```

The SIFT/RootSIFT variants work on the base install. The `*_VGG16*` variants build a
`DeepConvFeature`, so they need the `nn` extra: `pip install "pyvisim[nn]"`.

All clustering models were trained with `k=256`. The choice of `k` was made arbitrarily
based on the paper <sup>[5](#references)</sup>, where the authors tested with `k=32`, `64`, `128`, `256`, `512`, and so on.
Since higher values would take too long, I chose `k=256` as a balance between performance and computational cost.
Variants ending in `_PCA` reduce the feature dimensions by half with PCA before clustering.

> [!CAUTION]
> **Deprecated:** the old `weights=KMeansWeights.X` / `weights=GMMWeights.X` constructor
> argument is deprecated and will be removed in `1.0.0`. Use `from_pretrained()` with the
> `PretrainedVLAD`/`PretrainedFisher` enums (or `load_from_disk()` with a `.encoder` file)
> instead. The enums above load the exact same trained models.

### VLAD encoders (`PretrainedVLAD`)

Loaded with `VLADEncoder.from_pretrained(...)`.

| Member                        | Feature Extractor       | PCA Applied | Feature Dimensions |
|-------------------------------|-------------------------|-------------|--------------------|
| `OXFORD102_K256_VGG16_PCA`    | Last Conv Layer (VGG16) | Yes         | 257                |
| `OXFORD102_K256_VGG16`        | Last Conv Layer (VGG16) | No          | 514                |
| `OXFORD102_K256_ROOTSIFT_PCA` | RootSIFT features       | Yes         | 64                 |
| `OXFORD102_K256_ROOTSIFT`     | RootSIFT features       | No          | 128                |
| `OXFORD102_K256_SIFT_PCA`     | SIFT features           | Yes         | 64                 |
| `OXFORD102_K256_SIFT`         | SIFT features           | No          | 128                |

### Fisher Vector encoders (`PretrainedFisher`)

Loaded with `FisherVectorEncoder.from_pretrained(...)`.

| Member                        | Feature Extractor       | PCA Applied | Feature Dimensions |
|-------------------------------|-------------------------|-------------|--------------------|
| `OXFORD102_K256_VGG16_PCA`    | Last Conv Layer (VGG16) | Yes         | 257                |
| `OXFORD102_K256_VGG16`        | Last Conv Layer (VGG16) | No          | 514                |
| `OXFORD102_K256_ROOTSIFT_PCA` | RootSIFT features       | Yes         | 64                 |
| `OXFORD102_K256_ROOTSIFT`     | RootSIFT features       | No          | 128                |
| `OXFORD102_K256_SIFT_PCA`     | SIFT features           | Yes         | 64                 |
| `OXFORD102_K256_SIFT`         | SIFT features           | No          | 128                |

### Notes
1. **Feature Extraction**:
   - **Deep Features (VGG16)**: Feature maps from the last convolutional layer of VGG16. At each spatial location,
   the relative x and y coordinates are concatenated to the feature vector, resulting in `512 + 2 = 514` dimensions <sup>[6](#references)</sup>.
   - **SIFT**: Scale-Invariant Feature Transform descriptors, which was the original feature used for VLAD and
    Fisher Vector encoding <sup>[5](#references)</sup>.
   - **RootSIFT**: A variant of SIFT with `Hellinger kernel normalization`<sup>[4](#references)</sup>.
2. **Dimensionality Reduction**:
   - Models with `_PCA` in their names apply PCA to reduce the feature dimensions to by half.
   - The clustering models will learn from the transformed features after PCA is applied.

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

- With `v1.0.0`, remove the deprecated `weights` constructor argument and the `_CLUSTERING_TO_PCA_MAPPING` internal variable, since they are no longer needed with the new `from_pretrained()` API.
- Implement the **siamese network**.
- Add **tensor sketch approximation** and **mutual information** analysis for Fisher Vector, according to this
paper by Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng <sup>[1](#references)</sup>
- Add support for **vision transformers** for the `DeepConvFeature` class.

You are welcome to implement any of these features or suggest new ones!

## License
This project is licensed under the terms of the MIT license.

## References

[1] Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng, "Refining Deep Convolutional Features for
Improving Fine-Grained Image Recognition," EURASIP Journal on Image and Video Processing, 2017. \
[2] Relja Arandjelović and Andrew Zisserman, 'All About VLAD', Department of Engineering Science, University of Oxford. \
[3] E. Spyromitros-Xioufis, S. Papadopoulos, I. Kompatsiaris, G. Tsoumakas, and I. Vlahavas, "An Empirical Study on the
Combination of SURF Features with VLAD Vectors for Image Search," Informatics and Telematics Institute, Center for Research and
Technology Hellas, Thessaloniki, Greece; Department of Informatics, Aristotle University of Thessaloniki, Greece. \
[4] Relja Arandjelović and Andrew Zisserman, "Three things everyone should know to improve object retrieval," Department of  
Engineering Science, University of Oxford. \
[5] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local
Image Descriptors into Compact Codes," IEEE. \
[6] Liangliang Wang and Deepu Rajan, "An Image Similarity Descriptor for Classification Tasks," J. Vis. Commun.
Image R., vol. 71, pp. 102847, 2020. \
[7] [Oxford Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
