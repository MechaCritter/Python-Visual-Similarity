"""
Feature-extraction backbones and the shared base for the networks built from a
backbone and a projection head.
"""

import abc
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from PIL import Image

from ..lazy_import import OptionalImport
from ..typing import FloatNumpyArray, ImageInput, MatLike
from ..utils.image_utils import iter_images
from ._base import NeuralImageEmbedder

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    from ..utils.torch_utils import resolve_device

_torch_import.check()


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone with its final fully-connected layer removed.

    The classification head is stripped so the network outputs raw features;
    ``output_dim`` reports their dimensionality (512 for ResNet-18).

    :param pretrained: Whether to load the default ImageNet-pretrained weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.output_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts flattened backbone features.

        :param x: Input image tensor of shape (batch, channels, H, W).
        :return: Feature tensor of shape (batch, output_dim).
        """
        x = self.features(x)
        return x.flatten(1)


_TRANSFORM_REGISTRY: dict[str, Callable[[], "transforms.Compose"]] = {
    # Source: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    "resnet18": lambda: transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # this also rescales the pixel values to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    ),
}


def get_transform(backbone: str) -> "transforms.Compose":
    """
    Returns the preprocessing transform the given backbone was trained with.

    :param backbone: Name of the backbone network, e.g. ``"resnet18"``.
    :return: A fresh torchvision transform for that backbone.
    :raises ValueError: If no transform is registered for ``backbone``.
    """
    if backbone not in _TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}; no preprocessing transform is "
            f"registered for it. Supported backbones: "
            f"{', '.join(repr(name) for name in _TRANSFORM_REGISTRY)}."
        )
    return _TRANSFORM_REGISTRY[backbone]()


class BackboneWithHead(NeuralImageEmbedder):
    """
    Abstract base for embedders made of a feature backbone and a projection head.

    Every image goes through the same two stages: a pretrained
    feature-extraction ``backbone`` and a projection ``head`` that maps its
    features into the embedding space.

    NOTE
    ----
    - The serialization stores the transform as its ``repr`` because `transforms.Compose`
    is stateful and hence not JSON-safe. Upon deserialization, if the transform does
    not match the one the network was built with, a warning is issued.

    :param backbone: name of feature-extraction network.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param transform: processing transform applied to every input image. If
        ``None``, the preprocessing registered for the backbone is used. See
        :func:`~pyvisim.neural_networks.backbones.get_transform`.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the network from a checkpoint, set this
        to ``False`` to avoid downloading the weights again.
    :param similarity_func: Name of the built-in similarity metric used to score
        two embeddings. One of ``"cosine"`` (default), ``"euclidean"``, ``"l1"``
        or ``"manhattan"``.
    :raises ValueError: If ``embedding_dim`` is not a positive integer, if
        ``backbone`` is not a supported backbone name, or if ``similarity_func``
        is not a supported similarity metric.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        transform: transforms.Compose | None = None,
        pretrained_backbone: bool = True,
        similarity_func: str = "cosine",
    ):
        super().__init__(similarity_func=similarity_func)
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim}."
            )
        self._backbone_name = backbone
        self._embedding_dim = embedding_dim
        self._backbone = self._get_backbone(backbone, pretrained=pretrained_backbone)

        if transform is not None:
            self._transform = transform
        else:
            self._transform = get_transform(backbone)

        output_dim = cast(int, self._backbone.output_dim)
        self._head: torch.nn.Module = torch.nn.Linear(output_dim, embedding_dim)

    def _serialization_config(self) -> dict[str, Any]:
        return {
            "backbone": self._backbone_name,
            "embedding_dim": self._embedding_dim,
            "device": str(self.device),
            "transform": repr(self._transform),
        }

    @classmethod
    def _from_config(
        cls,
        config: dict[str, Any],
        *,
        transform: "transforms.Compose | None" = None,
        **kwargs: Any,
    ) -> "BackboneWithHead":
        cls._reject_unsupported_kwargs(kwargs)
        network = cls(
            backbone=config["backbone"],
            embedding_dim=config["embedding_dim"],
            transform=transform,
            pretrained_backbone=False,
        )

        if (
            serialized_transform := config.get("transform")
        ) is not None and serialized_transform != repr(network._transform):
            warnings.warn(
                f"The transform of this {type(network).__name__} differs from the one "
                "the saved network was built with, so the reloaded network will "
                f"produce different embeddings. Saved: {serialized_transform}; "
                f"current: {network._transform!r}. Pass the original transform back "
                f"with {type(network).__name__}.load_from_disk(path, transform=...) "
                "to reproduce them.",
                FutureWarning,
                stacklevel=2,
            )
        return network.to(resolve_device(config["device"]))

    @abc.abstractmethod
    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the shared-weight pass of the network on a preprocessed batch.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: Projected features of shape (batch, embedding_dim).
        """

    @staticmethod
    def _get_backbone(backbone: str, pretrained: bool) -> torch.nn.Module:
        """
        Returns the backbone network corresponding to the given name.

        :param backbone: Name of the backbone network.
        :param pretrained: Whether to use a pretrained backbone.
        :return: A PyTorch module implementing the backbone.
        :raises ValueError: If ``backbone`` is not recognized.
        """
        if backbone == "resnet18":
            return ResNetBackbone(pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}. Supported backbones: 'resnet18'."
            )

    def _preprocess(
        self,
        image: MatLike,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> torch.Tensor:
        """
        Preprocesses an image into a model-ready tensor.

        The input is rescaled from ``value_range`` to ``[0, 1]`` and routed
        through PIL; the resulting image is then passed through the network's
        ``transform``.

        :param image: Input image as a ``MatLike`` array.
        :param dims: Channel layout of the input, ``"HWC"`` (height x width x
            channels) or ``"CHW"`` (channels x height x width).
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: The transformed image tensor, of the shape the network's
            ``transform`` produces.
        :raises ValueError: If ``dims`` is not ``"HWC"`` or ``"CHW"``, or if
            ``value_range`` does not satisfy ``low < high``.
        """
        layout = dims.upper()
        if layout not in ("HWC", "CHW"):
            raise ValueError(f"dims must be 'HWC' or 'CHW', got {dims!r}.")
        lo, hi = value_range
        if hi <= lo:
            raise ValueError(f"value_range must satisfy low < high, got {value_range}.")

        arr = np.asarray(image, dtype=np.float32)
        if layout == "CHW":
            arr = arr.transpose(1, 2, 0)
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        pil_image = Image.fromarray(arr).convert("RGB")

        return cast(torch.Tensor, self._transform(pil_image))

    @torch.no_grad()
    def _embed_images(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> torch.Tensor:
        """
        Embeds one or more images into a batch of shared-weight pass outputs.

        All images are preprocessed, stacked into a single batch and passed
        through :meth:`_forward_once`. The model is switched to eval mode so
        that ``BatchNorm`` and ``Dropout`` behave correctly during inference,
        and the previous training state is restored afterwards so the training
        loop is not disrupted.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: ``(N, embedding_dim)`` tensor on the model's device.
        :raises ValueError: If ``images`` contains no image.
        """
        was_training = self.training
        self.eval()
        try:
            tensors = [
                self._preprocess(image)
                for image in iter_images(images, dims=dims, value_range=value_range)
            ]
            if not tensors:
                raise ValueError("Expected at least one image to embed, got none.")
            batch = torch.stack(tensors).to(self.device)
            return self._forward_once(batch)
        finally:
            if was_training:
                self.train()

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        embeddings = self._embed_images(images, dims=dims, value_range=value_range)
        return cast(FloatNumpyArray, embeddings.cpu().numpy())

    @property
    def backbone(self) -> torch.nn.Module:
        """The shared feature-extraction backbone."""
        return self._backbone

    @property
    def head(self) -> torch.nn.Module:
        """The projection head mapping backbone features to embeddings."""
        return self._head
