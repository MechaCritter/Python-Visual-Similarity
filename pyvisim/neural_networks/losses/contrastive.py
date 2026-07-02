from ...lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss for training Siamese embedding networks.

    The loss pulls embeddings of similar image pairs together and pushes
    dissimilar pairs apart up to a fixed ``margin``: a small embedding distance
    is rewarded for similar pairs (``label = 1``) while dissimilar pairs
    (``label = 0``) are only penalised while their distance is below the margin.

    References:
    ===========
    [1] Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction by
    Learning an Invariant Mapping. CVPR.

    [2] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.

    :param margin: Distance beyond which dissimilar pairs incur no loss.
    """

    def __init__(self, margin: float = 1.0) -> None:
        _torch_import.check()
        super().__init__()
        self.margin = margin

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean contrastive loss over a batch of embedding pairs.

        :param emb_a: Embeddings of the first images, shape (batch, dim).
        :param emb_b: Embeddings of the second images, shape (batch, dim).
        :param labels: Pair labels, ``1`` for similar and ``0`` for dissimilar
            pairs, shape (batch,).
        :return: The scalar loss averaged over the batch.
        """
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b)
        loss = 0.5 * (
            labels * distance.pow(2)
            + (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        )
        return loss.mean()
