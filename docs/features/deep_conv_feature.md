# DeepConvFeature

Extracts local descriptors from the convolutional feature map of a neural network
(default VGG16). Each spatial location in the chosen conv layer becomes one descriptor,
giving a CNN-based alternative to SIFT that plugs into the same embedders.

## `output_dim`

`output_dim` is the number of output channels of the selected conv layer. For VGG16's
last conv layer this is `512`.

## Selecting the layer

- `list_conv_layers()` enumerates the conv layers as `(index, name, module)`.
- `layer_index` chooses which to hook; `-1` (default) takes the last conv layer.
- `target_submodule` restricts the search to one named submodule of the model.

# TODO
- input range handling and batch processing; currently
  one image is processed per call.
