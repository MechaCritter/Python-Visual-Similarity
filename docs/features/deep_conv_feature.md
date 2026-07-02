# DeepConvFeature

File: [`_features.py`](../../pyvisim/features/_features.py)

> [!NOTE]
> Requires extra: `nn` (`pip install "pyvisim[nn]"`)


Extracts local descriptors from the convolutional feature map of a neural network
(default VGG16). Each spatial location in the chosen conv layer becomes one descriptor,
giving a CNN-based alternative to SIFT that plugs into the same encoders.

## Spatial encoding and `output_dim`

Convolutional features alone discard where in the image a descriptor came from.
Appending normalized coordinates re-injects coarse spatial information, which helps for
similarity tasks. When `spatial_encoding=True`, `output_dim` is the layer's output
channels `+ 2`; otherwise just the channel count. For VGG16's last conv layer this is
`512 + 2 = 514`.

## Selecting the layer

- `list_conv_layers()` enumerates the conv layers as `(index, name, module)`.
- `layer_index` chooses which to hook; `-1` (default) takes the last conv layer.
- `target_submodule` restricts the search to one named submodule of the model.

# TODO
- input range handling and batch processing; currently
  one image is processed per call.
