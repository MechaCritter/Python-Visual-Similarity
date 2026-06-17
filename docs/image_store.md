# pyvisim.image_store

File: [`pyvisim/image_store/__init__.py`](../pyvisim/image_store/__init__.py)
(Implementation: [`pyvisim/image_store/image_store.py`](../pyvisim/image_store/image_store.py))

`ImageEncodingMap` maps image file paths to their encoding vectors. It's a read-only
`Mapping`, so you use it like a dict: index by path, iterate, call `len`, `.values()`,
`dict(...)`, and so on.

## Lazy, then sticky

Nothing is encoded up front. The first time you ask for a path, the image is read,
encoded, and the vector is kept in an in-memory buffer. Ask for the same path again and
you get the buffered vector back instantly. The buffer is unbounded and never evicts on
its own, so once a vector is computed it stays put for the lifetime of the object.

```python
from pyvisim.encoders import VLADEncoder
from pyvisim.image_store import ImageEncodingMap

encoder = VLADEncoder(n_clusters=64)
encoder.learn(train_images)

store = ImageEncodingMap(encoder, ["a.jpg", "b.jpg", "c.jpg"])
vec = store["a.jpg"]        # reads + encodes "a.jpg" on this first access
vec_again = store["a.jpg"]  # served straight from the buffer, no re-encoding
```

You can also get one straight from an encoder or a pipeline via
[`generate_encoding_map`](encoders/base_encoder.md):

```python
store = encoder.generate_encoding_map(["a.jpg", "b.jpg", "c.jpg"])
```

Any object that satisfies the [`Encoder`](typing.md) protocol works here, so encoders and
`Pipeline` are both fair game.

## Freeing memory

Holding every encoding in memory is convenient but not free. When you're done with the
cached vectors, call `clear_buffer()` to drop them. The registered paths stay, so the next
access just re-encodes lazily.

```python
store.clear_buffer()  # buffer emptied; paths still known
store["a.jpg"]         # re-encoded on demand
```

## Saving and loading

`save_to_disk(path)` encodes everything (reusing whatever is already buffered) and writes
the full `path → encoding` mapping to an HDF5 file. `load_from_disk(path, encoder)` rebuilds
the map with those encodings loaded straight into the buffer, so reopening a saved store
involves no re-encoding.

```python
store.save_to_disk("encodings.h5")
restored = ImageEncodingMap.load_from_disk("encodings.h5", encoder)
```

A few things worth knowing:

- Pass `skip_errors=True` to `save_to_disk` to warn about and skip images that can't be
  read instead of aborting the whole save.
- Saving an empty store, or one where every image fails to encode, raises `ValueError`.
- `load_from_disk` warns if the encoder you pass doesn't match the one recorded in the
  file, since re-encoding new paths would then disagree with the stored vectors.
