# Lambda

File: [`_features.py`](../../pyvisim/features/_features.py)

`Lambda` wraps any user-defined function as a feature extractor, so you can plug a
custom descriptor into the encoders without writing a new `FeatureExtractorBase`
subclass.

## Usage

```python
from pyvisim.features import Lambda

extractor = Lambda(func=my_descriptor_fn, output_dim=64)
```

- `func` must take a single image (NumPy array), and return a
  `(N, output_dim)` array of descriptors.
- `output_dim` is supplied explicitly because, unlike SIFT or a CNN layer, an arbitrary
  function has no inspectable descriptor size. The encoders need this value to validate
  against PCA and the clustering model.
