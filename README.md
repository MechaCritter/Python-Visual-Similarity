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

`pyvisim` is a Python library for computing image similarities using image encoders
and neural networks.

📚 **Documentation**: <https://mechacritter.github.io/Python-Visual-Similarity/>

## Table of Contents

1. [Installation](#installation)
2. [Why **pyvisim**](#why-pyvisim)
4. [Contributing](#contributing)
5. [Get in Touch](#get-in-touch)
6. [TODO](#todo)
7. [License](#license)
8. [References](#references)

For a technical deep-dive into the library internals and the full API reference, see the
[hosted documentation](https://mechacritter.github.io/Python-Visual-Similarity/) (also
available in this repository as the [developer documentation](docs/overview.md)).

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
pip install pyvisim
# For deep learning features and the OxfordFlowerDataset
pip install "pyvisim[nn]"
# For image search feature
pip install "pyvisim[search]"
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

#### Example: Compute Similarity Score Using Vector of Locally Aggregated Descriptors (VLAD) <ref>[5]</ref>

```python
from pyvisim.encoders import VLADEncoder
from pyvisim.datasets import OxfordFlowerDataset  # needs "nn" extra: install with `pip install "pyvisim[nn]"`

# Load images from the Oxford Flower Dataset. Has to be NumPy Images!
dataset = OxfordFlowerDataset()
image1, *_  = dataset[0]
image2, *_ = dataset[1]

# Learn a visual vocabulary (RootSIFT features by default, k=256).
encoder = VLADEncoder(n_clusters=256)
encoder.learn(image for image, *_ in dataset)

# Compute the similarity score. By default, cosine similarity is used.
similarity_score = encoder.similarity_score(image1, image2)

print(f"Similarity Score: {similarity_score}")
```

By default the encoder uses cosine similarity. To use a different metric, pass
its name; `"cosine"`, `"euclidean"`, `"l1"` and `"manhattan"` are supported:

```python
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
   - Or skip the aggregation entirely and use `ClipEmbedder` (in `pyvisim.neural_networks`)
   for ready-made CLIP embeddings, loaded straight from OpenAI's official checkpoints.
   - The VGG16 deep-feature path (`DeepConvFeature`) and `ClipEmbedder` both need the `nn`
   extra: `pip install "pyvisim[nn]"`.

3. **Image Clustering**  
   - Cluster images based on their similarities to group them by category or content. An example and benchmarking
    can be found in [this notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/clustering_images_using_fv.ipynb).
   - Useful for organizing unlabeled data or generating pseudo-labels for further training.

4. **Pipeline for Combining Multiple Encoders**  
   - Chain various encoders in a single pipeline. An example can be found in [this notebook](https://github.com/MechaCritter/Python-Visual-Similarity-Examples/blob/master/notebooks/pipeline.ipynb).
   - Achieve more robust similarity metrics by blending different feature representations.

5. **Siamese Networks**  
   - Learn a similarity function directly from pairs of images with a Siamese network (needs the `nn` extra: `pip install "pyvisim[nn]"`).
   - Two variants are available: `ContrastiveSiameseNetwork` compares L2-normalized embeddings with a fixed
   metric and trains with the bundled `ContrastiveLoss` (Hadsell, Chopra & LeCun, 2006), while
   `BCESiameseNetwork` learns the comparison itself and returns the probability that two images
   show the same class (Koch et al., 2015). Both come with a ready-to-run training script:

     ```python
     from pyvisim.neural_networks import ContrastiveSiameseNetwork, BCESiameseNetwork

     model = ContrastiveSiameseNetwork(backbone="resnet18", embedding_dim=128)
     score = model.similarity_score(image1, image2)  # cosine similarity in [-1, 1]

     classifier = BCESiameseNetwork(backbone="resnet18", embedding_dim=128)
     probability = classifier.similarity_score(image1, image2)  # P(same class) in (0, 1)
     ```
     See the [neural networks docs](pyvisim/neural_networks/README.md) for more details.
   - Possible use cases include face recognition, signature verification, or any image-based identity matching.

### Notes

The local features the VLAD and Fisher Vector encoders aggregate:

- **RootSIFT** (the default): SIFT with `Hellinger kernel normalization` <sup>[4](#references)</sup>.
- **SIFT**: Scale-Invariant Feature Transform descriptors, the original feature used for VLAD and
  Fisher Vector encoding <sup>[5](#references)</sup>.
- **Deep Features (VGG16)**: Feature maps from the last convolutional layer of VGG16. At each spatial location,
  the relative x and y coordinates are concatenated to the feature vector, resulting in `512 + 2 = 514` dimensions <sup>[6](#references)</sup>.

Pass `pca_params` to reduce the feature dimensions before clustering; the clustering model
then learns from the transformed features.

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
