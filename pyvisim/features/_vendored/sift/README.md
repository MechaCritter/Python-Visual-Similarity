# SIFT Feature Extractor

The SIFT implementation is copied and adapted from scikit-image,
commit hash 5d91fed804ec8d9133118f02b3a785790df75934.
Source: https://github.com/scikit-image/scikit-image

The source of the files copied and modified:

- `pyvisim/features/sift/_vendored/sift.py`: `scikit-image/src/_skimage2/feature/sift.py`
- `pyvisim/features/sift/_vendored/dtype.py`: `scikit-image/src/_skimage2/util/dtype.py`
- `pyvisim/features/sift/_vendored/_warps.py`: `scikit-image/src/_skimage2/transform/_warps.py`
- `pyvisim/features/sift/_vendored/_utils.py`: `scikit-image/src/_skimage2/_shared/utils.py`
- `pyvisim/features/sift/_vendored/gaussian.py`: `scikit-image/src/_skimage2/filters/_gaussian.py`
- `pyvisim/features/sift/_vendored/_sift.pyx`: `scikit-image/src/_skimage2/feature/_sift.pyx`
- `pyvisim/features/sift/_vendored/_fused_numerics.pxd`: `scikit-image/src/_skimage2/_shared/fused_numerics.pxd`
- `tests/features/test_sift.py`:  `scikit-image/tests/skimage/feature/test_sift.py`

Copyright (c) 2009-2022 the scikit-image team
Licensed under BSD-3-Clause. See THIRD_PARTY_LICENSES/scikit-image.txt
for the full license text.
