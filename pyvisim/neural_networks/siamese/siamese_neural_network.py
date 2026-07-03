from typing import Any, cast

import numpy as np
from PIL import Image

from ..._base_classes import SimilarityMetric
from ..._utils import cosine_similarity
from ...encoders.utils import iter_images
from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput, MatLike, SimilarityFunc
from .backbones import ResNetBackbone

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class SiameseNeuralNetwork(torch.nn.Module, SimilarityMetric):
    """
    Siamese neural network for image similarity, proposed in
    `Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping`.

    #TODO: this is currently the Implementation originating from
    the paper [2]. Before the release, add the implementation from
    paper [1] as well, and allow the user to choose between the two.

    Two images are passed through the same shared-weight ``backbone`` and
    projection ``head`` to produce embeddings, which are L2-normalized so that
    cosine similarity reduces to a dot product. The network is trained so that
    similar images map to nearby embeddings and dissimilar images map far apart
    (see :class:`pyvisim.neural_networks.ContrastiveLoss`).

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
    :param similarity_func: Callable used to score two embeddings; defaults to
        cosine similarity.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone. See :meth:`_get_imagenet_transform`.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``SiameseNeuralNetwork`` from a checkpoint,
        set this to ``False`` to avoid downloading the weights again.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        similarity_func: SimilarityFunc = cosine_similarity,
        transform: transforms.Compose | None = None,
        device: str | torch.device = "cpu",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self._backbone = self._get_backbone(backbone, pretrained=pretrained_backbone)

        if transform is not None:
            self._transform = transform
        else:
            self._transform = self._get_imagenet_transform(backbone)

        output_dim = cast(int, self._backbone.output_dim)
        self._head: torch.nn.Module = torch.nn.Linear(output_dim, embedding_dim)
        self.similarity_func = similarity_func
        self.to(self.device)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for a batch of preprocessed images.

        The embeddings are unit-length, so cosine similarity between two of them
        equals their dot product.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: L2-normalized embeddings of shape (batch, embedding_dim).
        """
        features = self._backbone(x)
        embeddings = self._head(features)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings

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
        """
        arr = np.asarray(image, dtype=np.float32)
        if dims.upper() == "CHW":
            arr = arr.transpose(1, 2, 0)
        lo, hi = value_range
        arr = (arr - lo) / (hi - lo + 1e-8)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        pil_image = Image.fromarray(arr).convert("RGB")

        return cast(torch.Tensor, self._transform(pil_image))

    @torch.no_grad()
    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embeds one or more images into a batch of embedding vectors.

        All images are preprocessed, stacked into a single batch and passed
        through the network in one forward pass. The model is switched to eval
        mode so that ``BatchNorm`` and ``Dropout`` behave correctly during
        inference, and the previous training state is restored afterwards so
        the training loop is not disrupted.

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
        :return: ``(N, embedding_dim)`` array holding one L2-normalized
            embedding per input image.
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
                raise ValueError("Expected at least one image to encode, got none.")
            batch = torch.stack(tensors).to(self.device)
            embeddings = self.forward(batch)
            return cast(FloatNumpyArray, embeddings.cpu().numpy())
        finally:
            if was_training:
                self.train()

    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Computes the similarity matrix between two image batches.

        Both inputs are encoded with :meth:`encode` and the resulting embedding
        batches are scored with ``similarity_func``.

        :param image1: First (batch of) image(s) as ``MatLike``.
        :param image2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: Similarity matrix of shape ``(N, M)`` produced by
            ``similarity_func``.
        """
        embeddings1 = self.embed(image1, dims=dims, value_range=value_range)
        embeddings2 = self.embed(image2, dims=dims, value_range=value_range)
        return np.asarray(self.similarity_func(embeddings1, embeddings2))

    @property
    def backbone(self) -> torch.nn.Module:
        """The shared feature-extraction backbone (read-only)."""
        return self._backbone

    @property
    def head(self) -> torch.nn.Module:
        """The projection head mapping backbone features to embeddings."""
        return self._head
