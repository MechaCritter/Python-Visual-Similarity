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

    NOTE
    ----
    The reason for the margin being capped at 2 is that the embeddings are L2-normalized,
    under which the maximum possible distance between two embeddings is 2. Mathematically,
    when two vectors are L2-normalized, the L2 distance between them is actually
    just `cosine similarity` scaled to the range [0, 2]:

    ```
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * (u . v) = 1 + 1 - 2 * (u . v) = 2 * (1 - (u . v))
    ```

    Since `u` and `v` are unit vectors, `u . v` *is* the cosine similarity, so:

    ```
    ||u - v|| = sqrt(2 * (1 - cosine_similarity(u, v)))
    ```

    Cosine similarity ranges from -1 to 1. Plugging in the extremes shows the bound directly:

    - `cosine_similarity = 1`  (identical vectors) -> `||u - v||^2 = 2 * (1 - 1) = 0`
    - `cosine_similarity = -1` (opposite vectors)  -> `||u - v||^2 = 2 * (1 - (-1)) = 4`

    So `||u - v||^2` ranges from 0 to 4, and therefore the L2 distance `||u - v||`
    itself ranges from 0 to 2.

    References:
    ===========
    [1] Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction by
    Learning an Invariant Mapping. CVPR.

    [2] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks
    for One-shot Image Recognition. ICML Deep Learning Workshop.

    :param margin: Distance beyond which dissimilar pairs incur no loss.
    :raises ValueError: If ``margin`` is not strictly positive or is greater than 2.
    """

    def __init__(self, margin: float = 1.0) -> None:
        _torch_import.check()
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}.")
        if margin > 2:
            raise ValueError(
                f"margin should be <= 2 for L2-normalized embeddings, got {margin}."
            )
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
        self._validate_inputs(emb_a, emb_b, labels)
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b)
        loss = 0.5 * (
            labels * distance.pow(2)
            + (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        )
        return loss.mean()

    @staticmethod
    def _validate_inputs(
        emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Check shape consistency and binary labels.

        :raises ValueError: If one of following errors happen:
            - Embedding shapes differ.
            - Labels are not 1-dimensional.
            - Batch size mismatch between embeddings and labels.
            - Labels contain values other than 0 or 1.
        """
        if emb_a.shape != emb_b.shape:
            raise ValueError(
                f"Embedding shapes differ: {emb_a.shape} vs {emb_b.shape}."
            )
        if labels.dim() != 1:
            raise ValueError(
                f"labels must be 1-dimensional, got shape {tuple(labels.shape)}."
            )
        if labels.shape[0] != emb_a.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {labels.shape[0]} labels for "
                f"{emb_a.shape[0]} embedding pairs."
            )
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain only 0 and 1.")
