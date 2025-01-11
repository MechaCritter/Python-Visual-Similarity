<!-- Logo -->
<p align="center">
  <img src="res/images/logo.png" alt="pyvisim" width="1418" />
</p>

<!-- Added badges to convey project readiness/branding (example placeholders) -->
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-pre--release-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

# Welcome to `pyvisim`!

`pyvisim` is a Python library for computing image similarities using encoding methods such as Fisher Vectors, VLAD
and Siamese Networks. 

This project was made by me in cooperation with the __Lehrstuhl für Automatisierung und Informationssysteme__ 
at the Technical University of Munich.

## Table of Contents

1. [Why **pyvisim**](#why-pyvisim)
2. [Installation](#installation)
3. [Contributing](#contributing)
4. [Get in Touch](#get-in-touch)
5. [TODO](#todo)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)

## Why `pyvisim`?

`pyvisim` is designed to provide a simple and efficient way to compare images. The use-cases include:

1. **Image Retrieval**  
   Retrieve the top-k most similar images from a dataset.  
   - Use encoding methods like VLAD or Fisher Vectors to quickly find the most relevant matches. Please visit
   [this juptyer notebook](examples/vlad_fisher_with_vgg16_embeddings_image_retrieval.ipynb) for an example.
   - Example use: Building a fast image search engine for photo management software.

2. **Deep Learning Embeddings**  
   - Generate VLAD or Fisher vectors from neural network embeddings, e.g., VGG16 or other models.
   - Enhance your deep learning pipeline by leveraging traditional encoding methods on top of CNN features.

3. **Image Clustering**  
   - Cluster images based on their similarities to group them by category or content. An example and benchmarking
    can be found in [this notebook](examples/clustering_images_using_fv.ipynb).
   - Useful for organizing unlabeled data or generating pseudo-labels for further training.

4. **Pipeline for Combining Multiple Encoders**  
   - Chain various encoders in a single pipeline. An example can be found in [this notebook](examples/pipeline.ipynb).
   - Achieve more robust similarity metrics by blending different feature representations.

5. **Siamese Network (Coming Soon!)**  
   - Train a neural network to learn a similarity function directly from pairs/triples of images.  
   - Possible use cases include face recognition, signature verification, or any image-based identity matching.
   
## Installation

Currently, the package still needs to be tested and validated before being uploaded to PyPI. To use the library, you can
clone the repository and install the package locally:

```bash
git clone # TODO: Add the link to the repository
cd similarity_metrics_of_images
pip install .
```

All experiments in this project was made on the Oxford Flower Dataset <ref>[7]</ref>, for which I 
have created a custom dataset class. To use this class, import it as follows:

```python
from image_similarity.datasets import OxfordFlowerDataset
```
In order to replicate the experiments done in this project, follow the preprocessing steps in
the [datasets README](src/datasets/README.md).

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

- Implement proper **unit tests** to validate the functionality of the package before uploading to PyPI.
- Implement the **siamese network**.
- Add **tensor sketch approximation** and **mutual information** analysis for Fisher Vector, according to this
paper by Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng <sup>[1](#references)</sup>
- Add support for **vision transformers** for the `DeepConvFeature` class.

You are welcome to implement any of these features or suggest new ones!

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements

I would like to thank the __Lehrstuhl für Automatisierung und Informationssysteme__ at the Technical University of Munich
for their support, guidance and provision of computational resources for this project, without which this project would
not have been possible. I would also like to thank my supervisor, <add name>, for his guidance, as well as Dr. <add name>
for her valuable feedback and suggestions.

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

