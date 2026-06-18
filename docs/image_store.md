# pyvisim.image_store

File: [`pyvisim/image_store/__init__.py`](../pyvisim/image_store/__init__.py)
(Implementation: [`pyvisim/image_store/image_store.py`](../pyvisim/image_store/image_store.py))

`ImageEncodingMap` maps image file paths to their encoding vectors. It's a read-only
`Mapping`, so you use it like a dict: index by path, iterate, call `len`, `.values()`,
`dict(...)`, and so on.

## Encode images

You can encode a batch of images and get a mapping that maps
each path to its encoding vector (in form `{path: str → encoding: np.ndarray}` with the following code:

```python
from pyvisim.encoders import VLADEncoder
from pyvisim.image_store import ImageEncodingMap

encoder = VLADEncoder(n_clusters=64)
encoder.learn(train_images)

store = ImageEncodingMap(encoder, ["a.jpg", "b.jpg", "c.jpg"])  # encodes all three now
vec = store["a.jpg"]  # access the encoding.
```

A missing file raises `FileNotFoundError` and a file that isn't a valid image raises
`ValueError`. If you'd rather skip the bad ones, pass `skip_errors=True` and they're dropped
with a warning instead:

```python
store = ImageEncodingMap(encoder, ["a.jpg", "missing.jpg"], skip_errors=True)
# RuntimeWarning: Skipped 1 image(s) that could not be encoded.
```

You can also get a map straight from an encoder or a pipeline via
[`generate_encoding_map`](encoders/base_encoder.md):

```python
store = encoder.generate_encoding_map(["a.jpg", "b.jpg", "c.jpg"])
```

Any object that satisfies the [`Encoder`](typing.md) protocol works here, so encoders and
`Pipeline` are both fair game.

## Saving and loading

`save_to_disk(path)` writes the `path → encoding` mapping to a
[safetensors](https://github.com/huggingface/safetensors) file: the stacked encoding matrix
is stored as a single tensor, and the paths, encoder name, and format version live in the
file metadata. `load_from_disk(path, encoder)` reads those vectors back.

```python
store.save_to_disk("encodings.safetensors")
restored = ImageEncodingMap.load_from_disk("encodings.safetensors", encoder)
```
