from typing import cast

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

#: Standard ImageNet preprocessing applied to every input image.
_IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SiameseNeuralNetwork(torch.nn.Module, SimilarityMetric):
    """
    Siamese neural network for image similarity.

    Two images are passed through the same shared-weight ``backbone`` and
    projection ``head`` to produce embeddings, which are L2-normalized so that
    cosine similarity reduces to a dot product. The network is trained so that
    similar images map to nearby embeddings and dissimilar images map far apart
    (see :class:`pyvisim.neural_networks.ContrastiveLoss`).

    References:
    ===========
    [1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.

    :param backbone: Feature-extraction network exposing an ``output_dim``
        attribute (e.g. a ResNet-18 with its final layer removed). Defaults to a
        pretrained :class:`ResNetBackbone` when ``None``.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param similarity_func: Callable used to score two embeddings; defaults to
        cosine similarity.
    :param device: Device on which the model is placed.
    """

    def __init__(
        self,
        backbone: torch.nn.Module | None = None,
        embedding_dim: int = 128,
        similarity_func: SimilarityFunc = cosine_similarity,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self._backbone = backbone if backbone is not None else ResNetBackbone()
        output_dim = cast(int, self._backbone.output_dim)
        self._head: torch.nn.Module = torch.nn.Linear(output_dim, embedding_dim)
        self.similarity_func = similarity_func
        self.to(self.device)

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

        return cast(torch.Tensor, _IMAGENET_TRANSFORM(pil_image))

    @torch.no_grad()
    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Encodes one or more images into a batch of embedding vectors.

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
        embeddings1 = self.encode(image1, dims=dims, value_range=value_range)
        embeddings2 = self.encode(image2, dims=dims, value_range=value_range)
        return np.asarray(self.similarity_func(embeddings1, embeddings2))

    @property
    def head(self) -> torch.nn.Module:
        """The projection head mapping backbone features to embeddings."""
        return self._head

    @head.setter
    def head(self, new_head: torch.nn.Module) -> None:
        """
        Replaces the projection head.

        :param new_head: The replacement head. If it exposes an ``in_features``
            attribute (e.g. :class:`torch.nn.Linear`), it must match the
            backbone's ``output_dim``.
        :raises ValueError: If ``new_head`` is not a :class:`torch.nn.Module` or
            its ``in_features`` does not match the backbone's ``output_dim``.
        """
        if not isinstance(new_head, torch.nn.Module):
            raise ValueError(
                f"Expected new_head to be an instance of torch.nn.Module, got {type(new_head)}"
            )

        in_features = getattr(new_head, "in_features", None)
        if in_features is not None and in_features != self._backbone.output_dim:
            raise ValueError(
                f"Expected new_head to have in_features equal to {self._backbone.output_dim}, got {new_head.in_features}"
            )
        self._head = new_head
