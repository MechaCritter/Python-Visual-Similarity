# arc42: pyvisim

Software architecture of the library as a whole. This document is for
developers and is not part of the published documentation. Each module has its
own `arc42.md` next to its pages, and this one covers what is shared between
them.

## Building block view

- [Typing](typing/arc42.md): the public types and the protocols the components
  are written against.
- [Distance](distance/arc42.md): the distance metrics that compare embeddings.
- [Dense](dense/arc42.md): what the dense metrics share.
- [Structural](dense/structural/arc42.md): SSIM and MSSSIM.
- [Pixelwise](dense/pixelwise/arc42.md): PSNR.
- [Classic](classic/arc42.md): classical embedding methods pre deep learning
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

Pickling is avoided on purpose due to the risks of pickled files containing malicious
objects. Arrays are written as
[safetensors](https://github.com/huggingface/safetensors), and the structure plus the
scalars are stored as one JSON blob in the file metadata. A class-name registry then
maps a file back to the class that wrote it.

### A serializable class owns its file format under dunder names

The [SerializerMixin](pyvisim/serialization/mixin.py) defines the serialization
and deserialization interface for serializable classes (which are **almost all
classes** in `pyvisim`).

For each class, following class attributes must be defined:

- `__file_format__`: the file suffix, appended to the filename upon saving
to disk.
- `__metadata_key__`: upon saving to disk, the `safetensors` file will contain this
metadata key. If the key is missing (for example, loading an arbitrary `safetensors` file
that does not belong to this library), the load will be rejected.
- `__class_key__`: the key that maps to the class name in the `safetensors` file metadata.
- `__format_version__`: whenever the serialization interface is updated, this version
number is incremented by 1.
- `__state_keys__`: keys that describe which attributes of the class will flow into
the serialized file.

An example is provided below:

```python
from pyvisim.serialization import SerializerMixin

class Embedder(SerializerMixin):
    __file_format__ = ".safetensors"
    __metadata_key__ = "pyvisim_metadata"
    __class_key__ = "pyvisim_class"
    __format_version__ = 1
    __state_keys__ = ["similarity_func", "embedding_dim"]

    def __init__(self, similarity_func, embedding_dim):
        self.similarity_func = similarity_func
        self.embedding_dim = embedding_dim

    @classmethod
    def from_dict(cls, state):
        instance = cls(
            similarity_func=state["similarity_func"],
            embedding_dim=state["embedding_dim"]
        )
        return instance

    def _state(self):
        return {
          "similarity_func": self.similarity_func,
          "embedding_dim": self.embedding_dim
        }
```

Now, instantiate the object and save it to disk.

```python
embedder = Embedder(similarity_func="cosine", embedding_dim=128)
embedder.save_to_disk("my_embedder")
```

Inspect the dictionary of the object after we have deserialized it:

```python
import json
from safetensors import safe_open


with safe_open("my_embedder.safetensors", framework="numpy") as f:
    meta = f.metadata()
    # NOTE: The metadata key matches __metadata_key__ defined in the class
    parsed = json.loads(meta["pyvisim_metadata"])
    print(json.dumps(parsed, indent=4))
```

Expected output. **Note** that the key `pyvisim_class` matches what is
defined in `__class_key__` of the class.

```json
{
    "format_version": 1,
    "pyvisim_class": "Embedder",
    "similarity_func": "cosine",
    "embedding_dim": 128
}
```

### Heavyweight dependencies are optional and imported lazily

`pyvisim` uses [Optional
Imports](https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/pyvisim/lazy_import) to
reduce the total install size for users who don't use deep learning features (which installs
`torch`). This way, import is attempted eagerly, and if the dependency is missing, the
resulting `ImportError` is caught and only re-raised once the code that needs
it is actually called. So if such users never touch a deep learning class/module,
no exception is raised.

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
