"""Mathematical and validation tests for :class:`TripletLoss`.

All loss values are hand-computed from the FaceNet hinge

.. math::

    L(a, p, n) = \\max\\big(0, \\, d(a, p) - d(a, n) + m\\big)

over the triplets the configured strategy mines from a labeled batch.

The embeddings below live on the x-axis of a 2-D space, so every pairwise
distance is a difference of two numbers and the mined triplets can be read
off by hand. The loss never normalizes its inputs, so these coordinates
reach the hinge unchanged.
"""

from __future__ import annotations

import pytest
import torch

from pyvisim.neural_networks.losses import TripletLoss

#: Batch used for the strategy comparison: three samples of class 0 at
#: x = 0, 4, 1 and one sample of class 1 at x = 2, which sits *between* the
#: samples of class 0 and therefore violates the margin for most triplets.
_MIXED_BATCH = torch.tensor([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
_MIXED_LABELS = torch.tensor([0, 0, 0, 1])

#: The margin every hand-computed expectation below uses.
_MARGIN = 0.5


# §1 constructor validation


def test_default_configuration_is_facenet() -> None:
    """The defaults are FaceNet's: margin 0.2, semi-hard mining, squared."""
    loss = TripletLoss()
    assert loss.margin == 0.2
    assert loss.mining == "semi_hard"
    assert loss.squared is True


def test_is_torch_module() -> None:
    """The loss is a ``torch.nn.Module`` and can live inside a model tree."""
    assert isinstance(TripletLoss(), torch.nn.Module)


@pytest.mark.parametrize("margin", [0.0, -0.5, -1.0])
def test_non_positive_margin_raises(margin: float) -> None:
    """A zero or negative margin raises ``ValueError``."""
    with pytest.raises(ValueError, match="margin must be > 0"):
        TripletLoss(margin=margin)


def test_unsupported_mining_strategy_raises() -> None:
    """An unrecognised mining strategy raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported mining strategy"):
        TripletLoss(mining="offline")


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_supported_mining_strategies_accepted(mining: str) -> None:
    """Every documented strategy is accepted and stored."""
    assert TripletLoss(mining=mining).mining == mining


# §2 mathematical correctness on a single mined triplet


#: One class-0 pair at x = 0 and x = 3 with a class-1 sample at x = 1, so
#: ``d(a, p) = 3`` is larger than both negative distances.
_SINGLE_TRIPLET = torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
_SINGLE_LABELS = torch.tensor([0, 0, 1])


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_single_violating_triplet(mining: str) -> None:
    """Every strategy mines the same two anchors of a three-image batch.

    Anchor ``x = 0`` pays ``3 - 1 + 0.5 = 2.5`` and anchor ``x = 3`` pays
    ``3 - 2 + 0.5 = 1.5``, so all of them average to 2.0. The class-1 sample
    has no positive and is never an anchor.
    """
    criterion = TripletLoss(margin=_MARGIN, mining=mining, squared=False)
    loss = criterion(_SINGLE_TRIPLET, _SINGLE_LABELS)
    assert loss.item() == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_separated_batch_has_zero_loss(mining: str) -> None:
    """A batch whose classes are farther apart than the margin costs nothing.

    The class-0 pair sits at ``x = 0`` and ``x = 1`` while the class-1 sample
    is at ``x = 3``, so ``d(a, p) + margin`` never reaches ``d(a, n)``.
    """
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    criterion = TripletLoss(margin=_MARGIN, mining=mining, squared=False)
    loss = criterion(embeddings, torch.tensor([0, 0, 1]))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# §3 the strategies differ on the same batch


def test_batch_hard_averages_the_hardest_triplet_per_anchor() -> None:
    """``batch_hard`` mines one triplet per anchor and averages the three.

    Anchors ``x = 0`` and ``x = 4`` pay ``4 - 2 + 0.5``, anchor ``x = 1``
    pays ``3 - 1 + 0.5``, so the mean is 2.5. The class-1 sample has no
    positive and is skipped.
    """
    criterion = TripletLoss(margin=_MARGIN, mining="batch_hard", squared=False)
    loss = criterion(_MIXED_BATCH, _MIXED_LABELS)
    assert loss.item() == pytest.approx(2.5, abs=1e-6)


def test_batch_all_averages_only_the_violating_triplets() -> None:
    """``batch_all`` averages the five active hinges of the six valid triplets.

    The hinges are 2.5, 0 (satisfied), 2.5, 1.5, 0.5 and 2.5, and the
    satisfied one is left out of the mean, giving ``9.5 / 5 = 1.9``.
    """
    criterion = TripletLoss(margin=_MARGIN, mining="batch_all", squared=False)
    loss = criterion(_MIXED_BATCH, _MIXED_LABELS)
    assert loss.item() == pytest.approx(1.9, abs=1e-6)


def test_semi_hard_averages_over_all_positive_pairs() -> None:
    """``semi_hard`` averages over all six positive pairs, satisfied ones too.

    Only the pair ``(0, 1)`` of class 0 finds a negative farther than its
    positive, which zeroes its hinge. The remaining five fall back to the
    anchor's farthest negative and contribute 2.5, 2.5, 1.5, 0.5 and 2.5,
    so the mean over the six pairs is ``9.5 / 6``.
    """
    criterion = TripletLoss(margin=_MARGIN, mining="semi_hard", squared=False)
    loss = criterion(_MIXED_BATCH, _MIXED_LABELS)
    assert loss.item() == pytest.approx(9.5 / 6.0, abs=1e-6)


def test_semi_hard_prefers_the_closest_negative_beyond_the_positive() -> None:
    """A semi-hard negative replaces the fallback and can zero the hinge.

    Classes 0 and 1 sit at ``x = 0, 1`` and ``x = 2, 10``. The pair
    ``(x = 10, x = 2)`` is 8 apart and both of its negatives are farther, so
    the closest of them (9) is picked and ``8 - 9 + 0.5`` clamps to 0. Only
    the mirrored pair ``(x = 2, x = 10)`` has no semi-hard negative, falls
    back to its farthest negative (2) and pays ``8 - 2 + 0.5 = 6.5``.
    """
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    criterion = TripletLoss(margin=_MARGIN, mining="semi_hard", squared=False)
    loss = criterion(embeddings, torch.tensor([0, 0, 1, 1]))
    assert loss.item() == pytest.approx(6.5 / 4.0, abs=1e-6)


# §4 squared distances


def test_squared_distances_are_the_plain_ones_squared() -> None:
    """``squared=True`` feeds the hinge ``d^2`` instead of ``d``.

    On the mixed batch the hardest positives become 16, 16 and 9 and the
    closest negatives 4, 4 and 1, so the mean is ``33.5 / 3``.
    """
    criterion = TripletLoss(margin=_MARGIN, mining="batch_hard", squared=True)
    loss = criterion(_MIXED_BATCH, _MIXED_LABELS)
    assert loss.item() == pytest.approx(33.5 / 3.0, abs=1e-5)


def test_squared_distance_of_identical_embeddings_is_exactly_zero() -> None:
    """Coincident embeddings produce a distance of exactly 0, not a NaN.

    The square root has an infinite gradient at 0, so the implementation
    masks the diagonal and every other exact zero out.
    """
    embeddings = torch.zeros(4, 3, requires_grad=True)
    criterion = TripletLoss(margin=_MARGIN, mining="batch_all", squared=False)
    loss = criterion(embeddings, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


# §5 batches without a valid triplet


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_single_class_batch_has_zero_loss(mining: str) -> None:
    """A batch of one class has no negative, so nothing can be mined."""
    criterion = TripletLoss(margin=_MARGIN, mining=mining)
    loss = criterion(torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_batch_of_distinct_classes_has_zero_loss(mining: str) -> None:
    """A batch with one sample per class has no positive pair."""
    criterion = TripletLoss(margin=_MARGIN, mining=mining)
    loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 3]))
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_zero_loss_stays_connected_to_the_graph(mining: str) -> None:
    """An empty batch of triplets still back-propagates, with zero gradients.

    Returning a detached constant would break ``loss.backward()`` in the
    training loop as soon as one batch happens to hold a single class.
    """
    embeddings = torch.randn(4, 3, requires_grad=True)
    criterion = TripletLoss(margin=_MARGIN, mining=mining)
    loss = criterion(embeddings, torch.zeros(4, dtype=torch.long))

    assert loss.requires_grad
    loss.backward()
    assert embeddings.grad is not None
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_loss_returns_scalar_tensor() -> None:
    """The loss reduces the batch to a single 0-dimensional tensor."""
    loss = TripletLoss()(_MIXED_BATCH, _MIXED_LABELS)
    assert loss.shape == ()


# §6 gradients


def test_violating_triplet_pulls_the_anchor_towards_the_positive() -> None:
    """The gradient of the anchor points away from its positive.

    With ``L = d(a, p) - d(a, n) + m`` on the x-axis, the anchor at ``x = 0``
    has ``dL/da = (a - p)/d(a, p) - (a - n)/d(a, n) = -1 + 1 = 0`` in the
    degenerate collinear case, so the batch is spread over two dimensions:
    the positive is above the anchor and the negative to its right. A
    gradient step then has to move the anchor upwards, towards the positive.
    """
    embeddings = torch.tensor([[0.0, 0.0], [0.0, 3.0], [1.0, 0.0]], requires_grad=True)
    criterion = TripletLoss(margin=_MARGIN, mining="batch_hard", squared=False)
    criterion(embeddings, torch.tensor([0, 0, 1])).backward()

    assert embeddings.grad is not None
    # A descent step is -grad, so a negative y-gradient moves the anchor up.
    assert embeddings.grad[0, 1] < 0


def test_satisfied_batch_has_zero_gradient() -> None:
    """A batch that violates no margin contributes no gradient."""
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], requires_grad=True)
    criterion = TripletLoss(margin=_MARGIN, mining="batch_hard", squared=False)
    criterion(embeddings, torch.tensor([0, 0, 1])).backward()

    assert embeddings.grad is not None
    assert torch.allclose(embeddings.grad, torch.zeros_like(embeddings), atol=1e-6)


# §7 label handling


@pytest.mark.parametrize("mining", ["batch_all", "batch_hard", "semi_hard"])
def test_loss_depends_only_on_the_label_grouping(mining: str) -> None:
    """Renaming the classes leaves the mined loss unchanged."""
    criterion = TripletLoss(margin=_MARGIN, mining=mining, squared=False)
    renamed = torch.tensor([7, 7, 7, 3])
    assert criterion(_MIXED_BATCH, renamed).item() == pytest.approx(
        criterion(_MIXED_BATCH, _MIXED_LABELS).item(), abs=1e-6
    )


# §8 input validation


def test_non_two_dimensional_embeddings_raise() -> None:
    """A flat embedding tensor raises ``ValueError``."""
    with pytest.raises(ValueError, match="embeddings must be 2-dimensional"):
        TripletLoss()(torch.zeros(4), torch.tensor([0, 0, 1, 1]))


def test_non_one_dimensional_labels_raise() -> None:
    """Labels with a trailing dimension of 1 raise ``ValueError``."""
    with pytest.raises(ValueError, match="labels must be 1-dimensional"):
        TripletLoss()(torch.zeros(2, 3), torch.zeros(2, 1))


def test_batch_size_mismatch_raises() -> None:
    """A label count differing from the batch size raises ``ValueError``."""
    with pytest.raises(ValueError, match="Batch size mismatch"):
        TripletLoss()(torch.zeros(2, 3), torch.zeros(3))
