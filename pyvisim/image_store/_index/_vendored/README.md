# hnswlib

The approximate (HNSW) and exact (brute-force) search structures are copied from
hnswlib v0.9.0.
Source: https://github.com/nmslib/hnswlib

The source of the files copied and modified:

- `hnswlib/hnswlib/*.h`: `hnswlib/hnswlib/*.h`
- `hnswlib/python-bindings/bindings.cpp`: `hnswlib/python_bindings/bindings.cpp`

`bindings.cpp` carries three changes: the module is declared with
`PYBIND11_MODULE` instead of the `PYBIND11_PLUGIN` macro that pybind11 3.0
removed, `BFIndex` exposes the `space` and `dim` attributes that `Index` already
had, and both classes gained a `reset_index` method that frees the search
structure so `init_index` can build a new one over a replaced gallery. The
compile flags of the upstream `setup.py` are reproduced in this repository's
[`setup.py`](../../../../setup.py); `PYVISIM_NO_NATIVE=1` drops `-march=native`
for redistributable builds.

Copyright (c) Malkov, Yu A. and Yashunin, D. A. and hnswlib contributors.
Licensed under Apache-2.0. See THIRD_PARTY/hnswlib.txt for the full license
text.
