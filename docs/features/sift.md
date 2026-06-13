# SIFT

File: [`_features.py`](../../pyvisim/features/_features.py)

Scale-Invariant Feature Transform descriptors via OpenCV's `cv2.SIFT`. SIFT was the
original local descriptor used for VLAD and Fisher Vector encoding, so it is the
baseline handcrafted extractor here.

- `output_dim` is `128` (standard SIFT descriptor length).

For most uses prefer [`RootSIFT`](rootsift.md), which normalizes these descriptors and
usually improves retrieval at no extra cost.

## References

- D. G. Lowe. "Distinctive Image Features from Scale-Invariant Keypoints". In:
  International Journal of Computer Vision 60.2 (2004), pp. 91-110. issn: 1573-1405.
  doi: 10.1023/B:VISI.0000029664.99615.94.
  url: https://doi.org/10.1023/B:VISI.0000029664.99615.94.
