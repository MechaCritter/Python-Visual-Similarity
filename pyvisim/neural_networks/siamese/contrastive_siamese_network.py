import numpy as np

from ..._utils import get_similarity_func
from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput, SimilarityFunc
from ._base_siamese import SiameseNetworkBase

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class ContrastiveSiameseNetwork(SiameseNetworkBase):
    """
    Siamese network trained with a contrastive loss, proposed in
    `Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping`.

    This network "learns" the similarity metric directly. Two images are passed
    through the same shared-weight ``backbone`` and projection ``head`` to
    produce embeddings, which are L2-normalized so that cosine similarity
    reduces to a dot product. The network is trained so that similar images map
    to nearby embeddings and dissimilar images map far apart.

    `Contrastive loss` is used to train this network, which has the formula:

    .. math::
        L = \\frac{1}{2N} \\sum_{i=1}^{N} \\Bigl( y_i \\, D_i^2 + (1 - y_i) \\, \\max(0, m - D_i)^2 \\Bigr)

    References:
    ===========
    [1] Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction
    by Learning an Invariant Mapping. In Proceedings of the 2006 IEEE
    Computer Society Conference on Computer Vision and Pattern Recognition
    (CVPR), Vol. 2, 1735-1742. https://doi.org/10.1109/CVPR.2006.100

    :param backbone: name of feature-extraction network.
    :param embedding_dim: Dimensionality of the projected embedding space.
    :param similarity_func: Name of the built-in similarity metric used to score
        two embeddings. One of ``"cosine"``, ``"euclidean"``, ``"l1"``
        or ``"manhattan"``.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``ContrastiveSiameseNetwork`` from a
        checkpoint, set this to ``False`` to avoid downloading the weights again.
    :raises ValueError: If ``embedding_dim`` is not a positive integer, if
        ``backbone`` is not a supported backbone name, or if ``similarity_func``
        is not a supported similarity metric.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        similarity_func: str = "cosine",
        transform: transforms.Compose | None = None,
        device: str | torch.device = "cpu",
        pretrained_backbone: bool = True,
    ):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            transform=transform,
            pretrained_backbone=pretrained_backbone,
        )
        self._similarity_func: SimilarityFunc
        self.similarity_func = similarity_func
        self.to(torch.device(device))

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for a batch of preprocessed images.

        During training, call this once per branch of a pair and feed both
        embedding batches to the contrastive loss.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: L2-normalized embeddings of shape (batch, embedding_dim).
        """
        return self._forward_once(x)

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

        Both inputs are encoded with :meth:`embed` and the resulting embedding
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
    def similarity_func(self) -> SimilarityFunc:
        """The resolved similarity function callable."""
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, name: str) -> None:
        """
        Resolves and stores a built-in similarity metric by name.

        :param name: One of ``"cosine"``, ``"euclidean"``, ``"l1"`` or
            ``"manhattan"``.
        :raises ValueError: If ``name`` is not a supported similarity metric.
        """
        self._similarity_func = get_similarity_func(name)
