from typing import cast

import numpy as np
from PIL import Image

from .._base_classes import SimilarityMetric
from .._utils import cosine_similarity
from ..lazy_import import OptionalImport
from ..typing import FloatNumpyArray, MatLike, SimilarityFunc

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


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
        attribute (e.g. a ResNet-18 with its final layer removed).
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param similarity_func: Callable used to score two embeddings; defaults to
        cosine similarity.
    :param device: Device on which the model is placed.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        embedding_dim: int,
        similarity_func: SimilarityFunc = cosine_similarity,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self._backbone = backbone
        output_dim = cast(int, backbone.output_dim)
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

    def preprocess(
        self,
        image: MatLike | Image.Image,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> torch.Tensor:
        """
        Preprocesses an image into a model-ready tensor.

        Array inputs are rescaled from ``value_range`` to ``[0, 1]`` and routed
        through PIL; the image is then resized to 224x224 and normalized with
        the standard ImageNet statistics.

        :param image: Input image as a PIL image or ``MatLike`` array.
        :param dims: Channel layout for array input, ``"HWC"`` (height x width x
            channels) or ``"CHW"`` (channels x height x width).
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: A normalized image tensor of shape (channels, 224, 224).
        """
        if not isinstance(image, Image.Image):
            arr = np.asarray(image, dtype=np.float32)
            if dims.upper() == "CHW":
                arr = arr.transpose(1, 2, 0)
            lo, hi = value_range
            arr = (arr - lo) / (hi - lo + 1e-8)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            image = Image.fromarray(arr)

        image = image.convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        tensor: torch.Tensor = transform(image)
        return tensor

    @torch.no_grad()
    def encode(
        self,
        image: MatLike | Image.Image,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> FloatNumpyArray:
        """
        Encodes a single image into its embedding vector.

        The model is switched to eval mode so that ``BatchNorm`` and ``Dropout``
        behave correctly during inference, and the previous training state is
        restored afterwards so the training loop is not disrupted.

        :param image: Input image as a PIL image or ``MatLike`` array.
        :param dims: Channel layout of the input, ``"HWC"`` or ``"CHW"``.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: The image embedding as a 1-D NumPy array.
        """
        was_training = self.training
        self.eval()
        try:
            x = self.preprocess(image, dims=dims, value_range=value_range)
            x = x.unsqueeze(0).to(self.device)
            embedding = self.forward(x)
            return cast(FloatNumpyArray, embedding.cpu().numpy().squeeze())
        finally:
            if was_training:
                self.train()

    def similarity_score(  # type: ignore[override]
        self,
        image_a: MatLike | Image.Image,
        image_b: MatLike | Image.Image,
        dims_a: str = "HWC",
        dims_b: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> float:
        """
        Computes the similarity between two images.

        .. note::
            This overrides :meth:`SimilarityMetric.similarity_score` with a
            pairwise, scalar-returning signature: it accepts a separate channel
            layout for each image and returns a single ``float`` score rather
            than a similarity matrix.

        :param image_a: First image as a PIL image or ``MatLike`` array.
        :param image_b: Second image as a PIL image or ``MatLike`` array.
        :param dims_a: Channel layout of ``image_a``, ``"HWC"`` or ``"CHW"``.
        :param dims_b: Channel layout of ``image_b``, ``"HWC"`` or ``"CHW"``.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: The similarity score produced by ``similarity_func``.
        """
        emb_a = self.encode(image=image_a, dims=dims_a, value_range=value_range)
        emb_b = self.encode(image=image_b, dims=dims_b, value_range=value_range)
        score = self.similarity_func(emb_a, emb_b)
        return float(score.item())

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
