"""Mathematical and validation tests for :class:`ContrastiveLoss`.

All loss values are hand-computed from the Hadsell/Chopra/LeCun formulation

.. math::

    L = 0.5 \\cdot \\big( y \\cdot d^2 + (1 - y) \\cdot \\max(m - d, 0)^2 \\big)

averaged over the batch, where ``d`` is the pairwise L2 distance, ``y = 1``
marks a similar pair and ``m`` is the margin.

Note: ``torch.nn.functional.pairwise_distance`` adds ``eps = 1e-6`` inside the
norm, so "exact" expectations carry an error of about ``1e-6`` and are asserted
with tolerances of ``1e-4``.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyvisim.neural_networks.losses import ContrastiveLoss

# §1 constructor validation


def test_default_margin_is_one() -> None:
    """The default margin equals 1.0."""
    assert ContrastiveLoss().margin == 1.0


def test_is_torch_module() -> None:
    """The loss is a ``torch.nn.Module`` and can live inside a model tree."""
    assert isinstance(ContrastiveLoss(), torch.nn.Module)


@pytest.mark.parametrize("margin", [0.0, -0.5, -1.0])
def test_non_positive_margin_raises(margin: float) -> None:
    """A zero or negative margin raises ``ValueError``."""
    with pytest.raises(ValueError, match="margin must be > 0"):
        ContrastiveLoss(margin=margin)


@pytest.mark.parametrize("margin", [2.0001, 2.5, 100.0])
def test_margin_above_two_raises(margin: float) -> None:
    """A margin above 2 is impossible for L2-normalized embeddings."""
    with pytest.raises(ValueError, match="margin should be <= 2"):
        ContrastiveLoss(margin=margin)


@pytest.mark.parametrize("margin", [0.5, 1.0, 2.0])
def test_valid_margins_accepted(margin: float) -> None:
    """Margins in ``(0, 2]`` (including the boundary 2) are accepted."""
    assert ContrastiveLoss(margin=margin).margin == margin


# §2 mathematical correctness on trivial pairs


def test_identical_pair_similar_has_zero_loss() -> None:
    """A similar pair with identical embeddings incurs no loss (d = 0)."""
    emb = torch.tensor([[1.0, 0.0]])
    loss = ContrastiveLoss(margin=1.0)(emb, emb.clone(), torch.tensor([1.0]))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_identical_pair_dissimilar_pays_full_margin() -> None:
    """A dissimilar pair at distance 0 costs ``0.5 * margin**2``."""
    emb = torch.tensor([[1.0, 0.0]])
    loss = ContrastiveLoss(margin=1.0)(emb, emb.clone(), torch.tensor([0.0]))
    assert loss.item() == pytest.approx(0.5, abs=1e-4)


def test_orthogonal_pair_similar() -> None:
    """A similar pair of orthogonal unit vectors costs ``0.5 * (sqrt(2))**2 = 1``."""
    emb_a = torch.tensor([[1.0, 0.0]])
    emb_b = torch.tensor([[0.0, 1.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([1.0]))
    assert loss.item() == pytest.approx(1.0, abs=1e-4)


def test_orthogonal_pair_dissimilar_beyond_margin_is_free() -> None:
    """A dissimilar pair at distance sqrt(2) > margin = 1 incurs no loss."""
    emb_a = torch.tensor([[1.0, 0.0]])
    emb_b = torch.tensor([[0.0, 1.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([0.0]))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_opposite_pair_similar_costs_two() -> None:
    """A similar pair of opposite unit vectors (d = 2) costs ``0.5 * 4 = 2``."""
    emb_a = torch.tensor([[1.0, 0.0]])
    emb_b = torch.tensor([[-1.0, 0.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([1.0]))
    assert loss.item() == pytest.approx(2.0, abs=1e-4)


def test_dissimilar_pair_inside_margin() -> None:
    """A dissimilar pair at d = 0.6 with margin 1 costs ``0.5 * 0.4**2 = 0.08``."""
    emb_a = torch.tensor([[0.0, 0.0]])
    emb_b = torch.tensor([[0.6, 0.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([0.0]))
    assert loss.item() == pytest.approx(0.08, abs=1e-4)


def test_orthogonal_pair_dissimilar_with_margin_two() -> None:
    """With margin 2, a dissimilar orthogonal pair costs ``0.5 * (2 - sqrt(2))**2``."""
    emb_a = torch.tensor([[1.0, 0.0]])
    emb_b = torch.tensor([[0.0, 1.0]])
    loss = ContrastiveLoss(margin=2.0)(emb_a, emb_b, torch.tensor([0.0]))
    expected = 0.5 * (2.0 - math.sqrt(2.0)) ** 2
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_batch_loss_is_mean_of_pair_losses() -> None:
    """The batch loss equals the mean of the four hand-computed pair losses.

    Pairs (margin 1): identical/similar -> 0, identical/dissimilar -> 0.5,
    orthogonal/similar -> 1, orthogonal/dissimilar -> 0. Mean = 0.375.
    """
    emb_a = torch.tensor([[1.0, 0.0]] * 4)
    emb_b = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, labels)
    assert loss.item() == pytest.approx(0.375, abs=1e-4)


def test_loss_is_symmetric_in_embeddings() -> None:
    """Swapping the two embedding batches leaves the loss unchanged.

    Only up to ``pairwise_distance``'s internal ``eps = 1e-6``, which is added
    to the (sign-flipped) difference and breaks exact symmetry.
    """
    torch.manual_seed(0)
    emb_a = torch.randn(8, 4)
    emb_b = torch.randn(8, 4)
    labels = torch.tensor([1.0, 0.0] * 4)
    criterion = ContrastiveLoss(margin=1.0)
    assert criterion(emb_a, emb_b, labels).item() == pytest.approx(
        criterion(emb_b, emb_a, labels).item(), abs=1e-5
    )


def test_loss_returns_scalar_tensor() -> None:
    """The loss reduces the batch to a single 0-dimensional tensor."""
    emb = torch.zeros(3, 2)
    loss = ContrastiveLoss()(emb, emb, torch.ones(3))
    assert loss.shape == ()


# §3 gradients


def test_gradient_of_similar_pair_pulls_embeddings_together() -> None:
    """For a similar pair, ``dL/d(emb_a)`` is ``emb_a - emb_b``.

    ``L = 0.5 * ||a - b||**2`` gives ``dL/da = a - b``, so a gradient step
    moves ``a`` toward ``b``.
    """
    emb_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
    emb_b = torch.tensor([[0.0, 1.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([1.0]))
    loss.backward()
    assert emb_a.grad is not None
    assert torch.allclose(emb_a.grad, emb_a.detach() - emb_b, atol=1e-4)


def test_no_gradient_for_dissimilar_pair_beyond_margin() -> None:
    """A dissimilar pair beyond the margin contributes zero gradient."""
    emb_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
    emb_b = torch.tensor([[-1.0, 0.0]])
    loss = ContrastiveLoss(margin=1.0)(emb_a, emb_b, torch.tensor([0.0]))
    loss.backward()
    assert emb_a.grad is not None
    assert torch.allclose(emb_a.grad, torch.zeros_like(emb_a), atol=1e-5)


# §4 input validation


def test_embedding_shape_mismatch_raises() -> None:
    """Different embedding shapes raise ``ValueError``."""
    with pytest.raises(ValueError, match="Embedding shapes differ"):
        ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2))


def test_batch_size_mismatch_raises() -> None:
    """A label count differing from the batch size raises ``ValueError``."""
    with pytest.raises(ValueError, match="Batch size mismatch"):
        ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(3))


def test_non_one_dimensional_labels_raise() -> None:
    """Labels with a trailing dimension of 1 raise ``ValueError``."""
    with pytest.raises(ValueError, match="labels must be 1-dimensional"):
        ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 1))


@pytest.mark.parametrize("bad_value", [0.5, -1.0, 2.0])
def test_non_binary_labels_raise(bad_value: float) -> None:
    """Labels other than 0 and 1 raise ``ValueError``."""
    labels = torch.tensor([1.0, bad_value])
    with pytest.raises(ValueError, match="only 0 and 1"):
        ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(2, 3), labels)


def test_integer_binary_labels_accepted() -> None:
    """Integer 0/1 labels pass validation."""
    loss = ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
