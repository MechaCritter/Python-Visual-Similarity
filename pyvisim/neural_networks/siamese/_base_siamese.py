import abc
import warnings
from typing import Any, cast

import numpy as np
from PIL import Image

from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput, MatLike
from ...utils.image_utils import iter_images
from .._base import NeuralImageEmbedder
from .backbones import ResNetBackbone, get_transform

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

    from ...utils.torch_utils import resolve_device

_torch_import.check()


class SiameseNetworkBase(NeuralImageEmbedder):
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
        ``None``, the preprocessing registered for the backbone is used. See
        :func:`~pyvisim.neural_networks.siamese.backbones.get_transform`.
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

        self._has_custom_transform = transform is not None
        if transform is not None:
            self._transform = transform
        else:
            self._transform = get_transform(backbone)

        output_dim = cast(int, self._backbone.output_dim)
        self._head: torch.nn.Module = torch.nn.Linear(output_dim, embedding_dim)

    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the architecture description needed to rebuild this network.

        A custom ``transform`` is *not* serialised: torchvision transforms are
        arbitrary callables with no portable description. Reloading a network
        that was built with one therefore falls back to the preprocessing
        registered for its backbone, which is worth a warning since it changes
        the embeddings.

        :return: A JSON-safe mapping of constructor arguments.
        """
        if self._has_custom_transform:
            warnings.warn(
                f"The custom transform of this {type(self).__name__} is not "
                "serialised; the reloaded network will use the preprocessing "
                f"registered for the {self._backbone_name} backbone and "
                "therefore produce different embeddings. Pass the transform "
                f"back with {type(self).__name__}.load_from_disk(path, "
                "transform=...) to reproduce them.",
                FutureWarning,
                stacklevel=2,
            )
        return {
            "backbone": self._backbone_name,
            "embedding_dim": self._embedding_dim,
            "device": str(self.device),
        }

    @classmethod
    def _from_config(
        cls,
        config: dict[str, Any],
        *,
        transform: "transforms.Compose | None" = None,
        **kwargs: Any,
    ) -> "SiameseNetworkBase":
        """
        Rebuilds the network described by a serialised configuration.

        The backbone is built without its ImageNet weights, which
        :meth:`~pyvisim.neural_networks.NeuralImageEmbedder.from_dict`
        overwrites with the serialised ones anyway.

        :param config: Mapping produced by :meth:`_serialization_config`.
        :param transform: The processing transform to give the rebuilt network.
            The serialised state cannot hold one, so a network that was built
            with a custom transform only gets it back when it is passed here;
            ``None`` (default) falls back to the preprocessing registered for
            the backbone.
        :return: A network whose architecture matches ``config``.
        :raises TypeError: If an unsupported keyword argument is passed.
        """
        cls._reject_unsupported_kwargs(kwargs)
        network = cls(
            backbone=config["backbone"],
            embedding_dim=config["embedding_dim"],
            transform=transform,
            pretrained_backbone=False,
        )
        return network.to(resolve_device(config["device"]))

    @abc.abstractmethod
    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs one branch of the Siamese network on a preprocessed batch.

        Both inputs of a pair go through this same shared-weight pass; it is
        also the embedding used by :meth:`embed`.

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
            ``transform`` produces (channels, 224, 224) for ``resnet18``.
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
        Embeds one or more images into a batch of branch outputs.

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
