from typing import cast

from ...lazy_import import OptionalImport
from ...typing import FloatNumpyArray, ImageInput
from ._base_siamese import SiameseNetworkBase

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class PairwiseSiameseNetwork(SiameseNetworkBase):
    """
    Siamese network that classifies image pairs, proposed in
    `Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition`.

    Both images are passed through the same shared-weight ``backbone`` and
    projection ``head``; each branch output is squashed with a sigmoid into a
    feature vector ``h in (0, 1)^D`` (the paper's final fully-connected layer
    uses sigmoid units). The two branches are then combined by their
    component-wise L1 distance, and a single learned linear layer maps that
    distance vector to the probability of the pair showing the same class:

    .. math::

        p(x_1, x_2)
        = \\sigma\\Bigl(\\sum_{j} \\alpha_j \\, \\bigl| h_{1,j} - h_{2,j} \\bigr|
        + b\\Bigr)

    where the weights :math:`\\alpha_j` learn the importance of each feature
    dimension, so unlike :class:`ContrastiveSiameseNetwork` the comparison
    metric itself is trained. The network is a binary classifier over pairs and
    is trained with binary cross-entropy on labels ``1`` (same class) / ``0``
    (different class); :meth:`forward` returns raw logits so it composes with
    :class:`torch.nn.BCEWithLogitsLoss` in a numerically stable way.

    NOTE
    ----
    The score is a *learned probability*, not a geometric similarity: it is
    symmetric in its inputs (the L1 distance is), lives in ``(0, 1)``, and for
    two identical images equals ``sigmoid(b)`` -- the learned bias sets the
    operating point, so a perfect match does not score exactly ``1``.

    References:
    ===========
    [1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.
    https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

    :param backbone: name of feature-extraction network. Default: ``"resnet18"``.
    :param embedding_dim: Dimensionality of the twin feature vectors that the
        scoring layer compares.
    :param transform: processing transform applied to every input image. If
        ``None``, the default ImageNet preprocessing is used depending
        on the backbone. See :meth:`_get_imagenet_transform`.
    :param device: Device on which the model is placed.
    :param pretrained_backbone: Whether to use a backbone pretrained on
        ImageNet. If you are loading the ``PairwiseSiameseNetwork`` from a
        checkpoint, set this to ``False`` to avoid downloading the weights again.
    :raises ValueError: If ``embedding_dim`` is not a positive integer or if
        ``backbone`` is not a supported backbone name.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
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
        self._scorer: torch.nn.Module = torch.nn.Linear(embedding_dim, 1)
        self.to(torch.device(device))

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes sigmoid-activated feature vectors for a batch of images.

        Unlike the contrastive variant, the features are *not* L2-normalized;
        each component is squashed into ``(0, 1)`` so the component-wise L1
        distances fed to the scoring layer are bounded.

        :param x: Preprocessed image tensor of shape (batch, channels, H, W).
        :return: Feature tensor of shape (batch, embedding_dim) with values
            in ``(0, 1)``.
        """
        features = self._backbone(x)
        return torch.sigmoid(self._head(features))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes same-class logits for a batch of aligned image pairs.

        The i-th logit scores the pair ``(x1[i], x2[i])``; apply
        :func:`torch.sigmoid` to obtain probabilities, or feed the logits
        directly to :class:`torch.nn.BCEWithLogitsLoss` during training.

        :param x1: First preprocessed image batch, shape (batch, channels, H, W).
        :param x2: Second preprocessed image batch of the same shape.
        :return: Logit tensor of shape (batch,).
        :raises ValueError: If the two batches differ in shape.
        """
        if x1.shape != x2.shape:
            raise ValueError(
                f"Input batches must have the same shape, got "
                f"{tuple(x1.shape)} vs {tuple(x2.shape)}."
            )
        features1 = self._forward_once(x1)
        features2 = self._forward_once(x2)
        return self._score_distances(torch.abs(features1 - features2))

    def _score_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Maps component-wise L1 distance vectors to same-class logits.

        :param distances: Tensor of shape (..., embedding_dim) holding
            ``|h_1 - h_2|`` for each pair.
        :return: Logit tensor of shape (...,).
        """
        return cast(torch.Tensor, self._scorer(distances).squeeze(-1))

    @torch.no_grad()
    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Computes the same-class probability matrix between two image batches.

        Both inputs are encoded by the shared branch and every cross pair is
        scored through the learned layer, so entry ``(i, j)`` is the model's
        probability that ``image1[i]`` and ``image2[j]`` show the same class.

        NOTE
        ----
        All ``N * M`` component-wise distance vectors are materialized at once,
        so memory grows with ``N * M * embedding_dim``. Score very large
        collections in chunks.

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
        :return: Probability matrix of shape ``(N, M)`` with values in
            ``(0, 1)``.
        """
        features1 = self._encode_images(image1, dims=dims, value_range=value_range)
        features2 = self._encode_images(image2, dims=dims, value_range=value_range)
        distances = torch.abs(features1.unsqueeze(1) - features2.unsqueeze(0))
        probabilities = torch.sigmoid(self._score_distances(distances))
        return cast(FloatNumpyArray, probabilities.cpu().numpy())

    @property
    def scorer(self) -> torch.nn.Module:
        """The learned layer mapping L1 distances to same-class logits."""
        return self._scorer
