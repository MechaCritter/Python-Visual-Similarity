# Compiled search structures

This package holds everything the vendored hnswlib sources are extended with.
The vendored files themselves are kept byte-identical with upstream (see
[`../_vendored/README.md`](../_vendored/README.md)), so the additions live here
instead of in them.

| File | Role |
| --- | --- |
| `upstream_index.h` | Includes the vendored bindings so their `Index` and `BFIndex` classes can be extended. |
| `pyvisim_index.h` | `PyvisimIndex` and `PyvisimBFIndex`, the two classes extending them. |
| `_hnswlib.cpp` | Declares the `_hnswlib` extension module and registers both classes on it. |
| `_hnswlib.pyi` | Type stubs for the classes the module exports. |

The extension is compiled from `_hnswlib.cpp` by the repository's
[`setup.py`](../../../../setup.py) and imported as
`pyvisim.image_store._index._bindings._hnswlib`. Rebuild it with
`make build-ext`.

## What the classes add

`PyvisimIndex` and `PyvisimBFIndex` inherit the search structures, the
serialisation and the query paths of the vendored classes unchanged, and add:

- `reset_index`, which frees the search structure so that `init_index` can
  build a new one over a replaced gallery. Upstream refuses to initialise a
  structure twice, which otherwise leaves rebuilding an index out of reach.
- `get_items` on the brute-force index, which copies stored vectors back out of
  it. Upstream exposes it on the HNSW index only, so anything that needs the
  vectors of a brute-force index has to keep a second copy of them.
- The `space` and `dim` attributes on the brute-force index, which upstream
  registers on the HNSW index only.

## The module entry point

The vendored `bindings.cpp` ends with an entry point declared through the
`PYBIND11_PLUGIN` macro, which names its module `hnswlib` and is deprecated
since pybind11 3.0. `upstream_index.h` redefines that macro across the include,
which turns the upstream entry point into an inline function nothing references
and leaves the `PYBIND11_MODULE` declaration in `_hnswlib.cpp` as the only entry
point of the extension.
