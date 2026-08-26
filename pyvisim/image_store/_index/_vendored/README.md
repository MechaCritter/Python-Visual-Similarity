# hnswlib

The approximate (HNSW) and exact (brute-force) search structures are copied from
hnswlib v0.9.0.
Source: https://github.com/nmslib/hnswlib

The source of the files copied, all of them unmodified:

- `hnswlib/hnswlib/*.h`: `hnswlib/hnswlib/*.h`
- `hnswlib/python-bindings/bindings.cpp`: `hnswlib/python_bindings/bindings.cpp`

The `Index` and `BFIndex` classes of `bindings.cpp` are extended, and the
extension module is declared, in [`../_bindings`](../_bindings/README.md). The
compile flags of the upstream `setup.py` are reproduced in this repository's
[`setup.py`](../../../../setup.py); `PYVISIM_NO_NATIVE=1` drops `-march=native`
for redistributable builds.

Copyright (c) Malkov, Yu A. and Yashunin, D. A. and hnswlib contributors.
Licensed under Apache-2.0. See THIRD_PARTY/hnswlib.txt for the full license
text.
