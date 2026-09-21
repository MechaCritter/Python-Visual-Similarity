# arc42: pyvisim

Software architecture of the library as a whole. This document is for
developers and is not part of the published documentation. Each module has its
own `arc42.md` next to its pages, and this one covers what is shared between
them.

## Building block view

- [Typing](typing/arc42.md): the public types and the protocols the components
  are written against.
- [Distance](distance/arc42.md): the distance metrics that compare embeddings.
- [Structural](structural/arc42.md): SSIM and MSSSIM.
- [Pixelwise](pixelwise/arc42.md): PSNR.
- [Classic](classic/arc42.md): embedding methods from before the deep learning
  era.
- [Image similarity retrieval](image_similarity_retrieval/arc42.md): the image
  store, its search indexes and the re-ranking of its results.
- [Features](features/arc42.md): image feature extractors.
- [Neural networks](neural_networks/arc42.md): Siamese networks, triplet
  networks and CLIP embedders.
- [Dataset](dataset/arc42.md): `torch` datasets.
- [Eval](eval/arc42.md): retrieval scoring functions.

The abstract base classes that every public class derives from live in
`pyvisim/base/`: `SimilarityMetric`, `FeatureExtractorBase`,
`ImageEmbedderBase`, `SerializableImageEmbedder` and `DenseMetricBase` (shared
by the dense metrics).

## Architecture decisions

### Serialization uses the safetensors `.embedder` format

Pickling is avoided on purpose, since one can never rule out that a pickle
file contains malicious objects. Arrays are written as
[safetensors](https://github.com/huggingface/safetensors), and the structure
plus the scalars are stored as one JSON blob in the file metadata. A class-name
registry then maps a file back to the class that wrote it.

`torch.save` and `torch.load` still work on the neural networks, the usual
PyTorch way.

### A serializable class owns its file format under dunder names

The class attributes `__file_format__`, `__metadata_key__`, `__class_key__`,
`__format_version__`, `__state_keys__` and `__compatibility_mapping__` describe
the file format, and `SerializerMixin` does the serialization and
deserialization based on them. Since these are plain class attributes, a
subclass inherits the contract and overrides only what differs. A new
serializable class therefore declares a version and its state keys instead of
repeating the file suffix, the metadata key and the class key.

The names carry leading *and* trailing double underscores. With two leading
underscores alone, Python would mangle the name inside every class body that
reads it. The trailing pair marks the attribute as part of the framework's
contract, so a user is not expected to set it.

### Heavyweight dependencies are optional and imported lazily

`pyvisim` offers heavyweight extras without forcing every user to install
them. This is done through [Optional
Imports](https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/pyvisim/lazy_import):
the import is attempted eagerly, and if the dependency is missing, the
resulting `ImportError` is caught and only re-raised once the code that needs
it is actually called.

The classical pipeline still *accepts* torch tensors when torch happens to be
installed, but it must not depend on torch. That is why `is_tensor` returns
`False` when torch is absent instead of raising.

### Vendored third-party code stays byte-identical to its source

Files under a `_vendored` folder are 1-to-1 copies of their original sources
and stay unchanged for the rest of their lifetime inside `pyvisim`. This way,
when the upstream changes, one only needs some text diff tool to compare the
changes and can simply overwrite the current files with the upstream ones.
Where a behavior change is necessary, a subclass or an overriding method is
added in a separate file.

## Risks and technical debt

- Add tensor sketch approximation and mutual information analysis for the
  Fisher Vector, according to the paper by Weixia Zhang, Jia Yan, Wenxuan Shi,
  Tianpeng Feng and Dexiang Deng.
- Add support for vision transformers to the `DeepConvFeature` class.
