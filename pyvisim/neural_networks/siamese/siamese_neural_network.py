import warnings

from ...lazy_import import OptionalImport
from .contrastive_siamese_network import ContrastiveSiameseNetwork

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()


class SiameseNeuralNetwork(ContrastiveSiameseNetwork):
    """
    Deprecated alias of :class:`ContrastiveSiameseNetwork`.

    The original ``SiameseNeuralNetwork`` implemented exactly the contrastive
    variant (Hadsell, Chopra & LeCun, 2006) and was renamed when the
    pair-classifying variant (:class:`PairwiseSiameseNetwork`, Koch et al.,
    2015) was added. Behaviour, constructor arguments and checkpoints are fully
    compatible with :class:`ContrastiveSiameseNetwork`.

    .. deprecated:: 0.9
        Use :class:`ContrastiveSiameseNetwork` instead; this alias will be
        removed in a future release.
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
        warnings.warn(
            "SiameseNeuralNetwork has been renamed to ContrastiveSiameseNetwork "
            "and will be removed in a future release; update your imports.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            similarity_func=similarity_func,
            transform=transform,
            device=device,
            pretrained_backbone=pretrained_backbone,
        )
