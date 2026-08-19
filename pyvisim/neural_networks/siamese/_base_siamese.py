import abc
from typing import Any, cast

import numpy as np
from PIL import Image

from ..._base_classes import SimilarityMetric
from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput, MatLike
from ...utils.image_utils import iter_images
from .backbones import ResNetBackbone

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class SiameseNetworkBase(torch.nn.Module, SimilarityMetric):
    """
    Abstract base for Siamese image-similarity networks.

    A Siamese network passes both inputs through the *same* shared-weight
    ``backbone`` and projection ``head``; concrete subclasses only differ in
    what they do with the two branch outputs:

    - :class:`ContrastiveSiameseNetwork` compares the branch embeddings with a
      fixed metric (e.g. cosine) and is trained with a contrastive loss
      (Hadsell, Chopra & LeCun, 2006).
    - :class:`BCESiameseNetwork` feeds the component-wise L1 distance of
      the branch features into a learned scoring layer that outputs the
      probability of the two images belonging to the same class
      (Koch, Zemel & Salakhutdinov, 2015). It has no similarity-preserving
      embedding space and therefore overrides :meth:`embed` to raise
      :class:`NotImplementedError`.

    Subclasses must implement :meth:`_forward_once` (the single-branch pass
    used by :meth:`embed`) and :meth:`similarity_score`, and must finish their
    ``__init__`` with ``self.to(torch.device(device))`` so that every submodule
    they register ends up on the requested device.

    References:
    ===========
    [1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.

    [2] Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping. In Proceedings of the 2006 IEEE
    Computer Society Conference on Computer Vision and Pattern Recognition
    (CVPR), Vol. 2, 1735-1742. https://doi.org/10.1109/CVPR.2006.100

    :param backbone: name of feature-extraction network. Default: ``"resnet18"``.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone. See :meth:`_get_imagenet_transform`.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the network from a checkpoint, set this
        to ``False`` to avoid downloading the weights again.
    :raises ValueError: If ``embedding_dim`` is not a positive integer or if
        ``backbone`` is not a supported backbone name.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        transform: transforms.Compose | None = None,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim}."
            )
        self._backbone = self._get_backbone(backbone, pretrained=pretrained_backbone)

        if transform is not None:
            self._transform = transform
        else:
            self._transform = self._get_imagenet_transform(backbone)

        output_dim = cast(int, self._backbone.output_dim)
        self._head: torch.nn.Module = torch.nn.Linear(output_dim, embedding_dim)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute, routing property assignments through the property.

        NOTE
        ----
        Since :class:`torch.nn.Module` overrides ``__setattr__`` and registers
        any ``torch.nn.Module`` value directly in ``self._modules``, assigning
        to a read-only property such as ``head`` would silently register an
        orphan submodule instead of failing. Routing assignments to class-level
        properties through ``property.__set__`` restores standard Python
        semantics. Hence, this override was necessary.

        :param name: Name of the attribute to set.
        :param value: Value to assign.
        :raises AttributeError: If ``name`` is a read-only property.
        """
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property):
            descriptor.__set__(self, value)
            return
        super().__setattr__(name, value)

    @abc.abstractmethod
    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs one branch of the Siamese network on a preprocessed batch.

        Both inputs of a pair go through this same shared-weight pass; it is
        also the encoding used by :meth:`embed`.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: Branch output of shape (batch, embedding_dim).
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

    @staticmethod
    def _get_imagenet_transform(backbone: str) -> transforms.Compose:
        """
        Returns the preprocessing transform for the given backbone.

        NOTE
        ----
        This assumes that the backbone trained on ImageNet.

        :param backbone: The name of the backbone network.
        :return: A torchvision transform that resizes, normalizes, and converts
            images to tensors.
        """
        # The backbone of ResNet-18 as it was trained on ImageNet. Reference:
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        if backbone == "resnet18":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),  # this also rescales the pixel values to [0, 1]
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}; no default ImageNet transform "
                f"is available. Supported backbones: 'resnet18'."
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
        through PIL; the image is then resized to 224x224 and normalized with
        the standard ImageNet statistics.

        :param image: Input image as a ``MatLike`` array.
        :param dims: Channel layout of the input, ``"HWC"`` (height x width x
            channels) or ``"CHW"`` (channels x height x width).
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: A normalized image tensor of shape (channels, 224, 224).
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
    def _encode_images(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> torch.Tensor:
        """
        Encodes one or more images into a batch of branch outputs.

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
        """
        Embeds one or more images into a batch of embedding vectors.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: ``(N, embedding_dim)`` array holding one embedding per input
            image.
        :raises ValueError: If ``images`` contains no image.
        """
        embeddings = self._encode_images(images, dims=dims, value_range=value_range)
        return cast(FloatNumpyArray, embeddings.cpu().numpy())

    @property
    def device(self) -> torch.device:
        """
        The device the model's parameters live on (read-only).

        Derived from the parameters themselves rather than cached, so it stays
        correct after the user moves the model with ``model.to(...)``.
        """
        return next(self.parameters()).device

    @property
    def backbone(self) -> torch.nn.Module:
        """The shared feature-extraction backbone (read-only)."""
        return self._backbone

    @property
    def head(self) -> torch.nn.Module:
        """The projection head mapping backbone features to embeddings."""
        return self._head
