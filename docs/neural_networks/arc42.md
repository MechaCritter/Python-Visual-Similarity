# arc42: neural_networks

Software architecture of `pyvisim.neural_networks`. This document is for
developers and is not part of the published documentation.

## Building block view

Every network in this module is a backbone plus a head. The backbone is the
pretrained convolutional network that turns an image into features. It is
built by name through `build_backbone`, and the same module provides the
ImageNet preprocessing the torchvision weights were trained with.
`pretrained=False` returns the bare architecture. Deserialization relies on
this, since the trained weights are loaded into it afterwards.

The Siamese and triplet networks realize their "branches" implicitly by weight
sharing, so each model holds a single backbone instance that all branches
share.

`ClipEmbedder` is the exception. It re-implements the CLIP image tower instead
of using a torchvision backbone, and it is an embedder, since it is not trained
inside this library.

`DeepConvFeature` has a backbone but no head. It is a feature extractor, so it
follows the contract of `pyvisim.features` and returns the local descriptors
of one image. It lives in this module because it reads those descriptors off
a backbone.

## Architecture decisions

### A serialized model stores its architecture and its weights separately

Serialization splits a model into a configuration that describes the
architecture (the backbone name, the head sizes, the CLIP variant and the
pretrained tag) and the learned weights, which are stored separately as the
model's `state_dict`. Reconstruction builds the architecture first and then
loads the weights into it, so no pretrained checkpoint has to be fetched to
restore a saved model.

### The losses are reimplemented on top of torch

Some distance modules are reimplemented with `torch` instead of being reused
from `pyvisim.distance`, so that the gradient flows through the loss in the
`forward` pass (for example in the [Triplet
Loss](https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/pyvisim/neural_networks/losses/triplet.py)).
