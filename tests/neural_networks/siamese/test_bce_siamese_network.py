"""Unit tests for :class:`pyvisim.neural_networks.BCESiameseNetwork`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import transforms

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.neural_networks import BCESiameseNetwork
from tests.neural_networks._stubs import (
    FlattenBackbone,
    MeanBackbone,
    build_stub_bce_model,
    install_head,
    install_scorer,
    make_identity_head,
    make_random_rgb_image,
    make_solid_rgb_image,
)

#: Embedding size of the shared ``bce_model`` fixture.
EMBEDDING_DIM = 4


def _sigmoid(x: float) -> float:
    """Reference logistic function for hand-computed expectations.

    :param x: The logit.
    :return: ``1 / (1 + exp(-x))``.
    """
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture
def bce_model() -> BCESiameseNetwork:
    torch.manual_seed(0)
    return build_stub_bce_model(MeanBackbone(), embedding_dim=EMBEDDING_DIM)


# §1 construction


def test_unknown_backbone_name_raises() -> None:
    """An unrecognised backbone name raises ``ValueError`` at construction."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        BCESiameseNetwork(backbone="alexnet")


def test_non_positive_embedding_dim_raises() -> None:
    """A non-positive embedding dimensionality is rejected."""
    with pytest.raises(ValueError, match="embedding_dim"):
        build_stub_bce_model(MeanBackbone(), embedding_dim=0)


def test_scorer_maps_embedding_to_single_logit(bce_model: BCESiameseNetwork) -> None:
    """The scoring layer is a ``Linear(embedding_dim, 1)``."""
    assert isinstance(bce_model.scorer, torch.nn.Linear)
    assert bce_model.scorer.in_features == EMBEDDING_DIM
    assert bce_model.scorer.out_features == 1


def test_head_input_matches_backbone_output_dim(bce_model: BCESiameseNetwork) -> None:
    """The head's ``in_features`` equals the backbone's ``output_dim``."""
    assert isinstance(bce_model.head, torch.nn.Linear)
    assert bce_model.head.in_features == MeanBackbone.output_dim


# §2 branch output: sigmoid features, not normalized embeddings


def test_embed_raises_not_implemented_error(bce_model: BCESiameseNetwork) -> None:
    """``embed`` is unsupported: the branch features are not an embedding."""
    with pytest.raises(
        NotImplementedError, match="does not learn to generate embeddings"
    ):
        bce_model.embed(make_random_rgb_image(seed=1))


def test_branch_applies_sigmoid_to_head_output() -> None:
    """A uniform image encodes to the constant vector ``sigmoid(v / 255)``.

    With a plain ``ToTensor`` transform, a 2x2 uniform image of value 51
    becomes 12 features of exactly ``0.2``; the identity head leaves them
    untouched and the branch sigmoid maps each to ``sigmoid(0.2)``.
    """
    model = build_stub_bce_model(
        FlattenBackbone(12),
        embedding_dim=12,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(12))
    features = model._encode_images(make_solid_rgb_image(value=51, size=2)).numpy()
    assert features.shape == (1, 12)
    assert np.allclose(features, _sigmoid(51.0 / 255.0), atol=1e-6)


def test_branch_output_is_bounded_but_not_unit_norm(
    bce_model: BCESiameseNetwork,
) -> None:
    """Branch features lie strictly in ``(0, 1)`` and are not L2-normalized."""
    features = bce_model._encode_images(
        [make_random_rgb_image(seed=1), make_random_rgb_image(seed=2)]
    ).numpy()
    assert np.all(features > 0.0)
    assert np.all(features < 1.0)
    assert not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-3)


# §3 forward: hand-computed pair logits


def test_forward_computes_weighted_l1_logits() -> None:
    """``forward`` returns ``sum(alpha * |h1 - h2|) + b`` per aligned pair.

    With an identity head the branch features are ``sigmoid(x)``. The first
    pair differs only in one component: ``|sigmoid(ln 3) - sigmoid(0)| =
    |0.75 - 0.5| = 0.25``, so with weights ``[4, 8]`` and bias ``-1`` its
    logit is ``4 * 0.25 - 1 = 0``. The second pair is identical, leaving
    only the bias ``-1``.
    """
    model = build_stub_bce_model(FlattenBackbone(2), embedding_dim=2)
    install_head(model, make_identity_head(2))
    install_scorer(model, weights=[4.0, 8.0], bias=-1.0)

    x1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    x2 = torch.tensor([[math.log(3.0), 0.0], [0.0, 0.0]])
    logits = model(x1, x2)

    assert logits.shape == (2,)
    assert torch.allclose(logits, torch.tensor([0.0, -1.0]), atol=1e-6)


def test_forward_is_symmetric_in_its_inputs() -> None:
    """Swapping the two branches leaves the logits unchanged (L1 distance)."""
    torch.manual_seed(0)
    model = build_stub_bce_model(FlattenBackbone(4), embedding_dim=4)
    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 4)
    assert torch.allclose(model(x1, x2), model(x2, x1), atol=1e-6)


def test_forward_rejects_mismatched_batch_shapes() -> None:
    """Batches of different shapes raise instead of broadcasting silently."""
    model = build_stub_bce_model(FlattenBackbone(2), embedding_dim=2)
    with pytest.raises(ValueError, match="same shape"):
        model(torch.zeros(1, 2), torch.zeros(3, 2))


# §4 similarity_score: probability matrix through the full public path


def test_similarity_score_matches_hand_computed_probability() -> None:
    """The full image-to-probability path reproduces the paper's formula.

    Two uniform images with pixel values 51 and 204 embed to the constant
    vectors ``sigmoid(0.2)`` and ``sigmoid(0.8)``. With unit weights and zero
    bias the logit is ``12 * (sigmoid(0.8) - sigmoid(0.2))`` and the score is
    its sigmoid.
    """
    model = build_stub_bce_model(
        FlattenBackbone(12),
        embedding_dim=12,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(12))
    install_scorer(model, weights=[1.0] * 12, bias=0.0)

    score = model.similarity_score(
        make_solid_rgb_image(value=51, size=2), make_solid_rgb_image(value=204, size=2)
    )

    logit = 12.0 * (_sigmoid(204.0 / 255.0) - _sigmoid(51.0 / 255.0))
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(_sigmoid(logit), abs=1e-5)


def test_identical_images_score_sigmoid_of_bias(bce_model: BCESiameseNetwork) -> None:
    """For identical inputs all L1 distances vanish, leaving ``sigmoid(b)``.

    A perfect match therefore scores the learned operating point, not 1.
    """
    install_scorer(bce_model, weights=[2.0, -3.0, 0.5, 1.0], bias=-1.0)
    image = make_random_rgb_image(seed=11)
    score = bce_model.similarity_score(image, image.copy())
    assert score[0, 0] == pytest.approx(_sigmoid(-1.0), abs=1e-6)


def test_similarity_score_uses_learned_scoring_layer(
    bce_model: BCESiameseNetwork,
) -> None:
    """The score equals the formula evaluated on the branch features.

    Reads the (randomly initialised) scoring parameters back and recomputes
    ``sigmoid(sum(alpha * |h1 - h2|) + b)`` from the branch outputs.
    """
    image1 = make_random_rgb_image(seed=1)
    image2 = make_random_rgb_image(seed=2)

    h1 = bce_model._encode_images(image1).numpy()[0]
    h2 = bce_model._encode_images(image2).numpy()[0]
    assert isinstance(bce_model.scorer, torch.nn.Linear)
    weights = bce_model.scorer.weight.detach().cpu().numpy()[0]
    bias = float(bce_model.scorer.bias.detach().cpu())
    expected = _sigmoid(float(np.sum(weights * np.abs(h1 - h2))) + bias)

    assert bce_model.similarity_score(image1, image2)[0, 0] == pytest.approx(
        expected, abs=1e-5
    )


def test_similarity_matrix_shape(bce_model: BCESiameseNetwork) -> None:
    """Two batches of sizes N and M produce an ``(N, M)`` probability matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = bce_model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)


def test_similarity_scores_are_probabilities(bce_model: BCESiameseNetwork) -> None:
    """Every entry of the score matrix lies strictly in ``(0, 1)``."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = bce_model.similarity_score(batch_a, batch_b)
    assert np.all(score > 0.0)
    assert np.all(score < 1.0)


def test_similarity_matrix_is_symmetric(bce_model: BCESiameseNetwork) -> None:
    """Swapping the inputs transposes the probability matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    forward_score = bce_model.similarity_score(batch_a, batch_b)
    backward_score = bce_model.similarity_score(batch_b, batch_a)
    assert np.allclose(forward_score, backward_score.T, atol=1e-6)


# §5 training integration


def test_bce_training_step_reduces_loss_on_separable_pairs() -> None:
    """A few BCE steps on trivially separable pairs drive the loss down.

    Positive pairs are identical vectors (zero L1 distance), negative pairs
    are far apart, so the scoring layer alone can separate them.
    """
    torch.manual_seed(0)
    model = build_stub_bce_model(FlattenBackbone(2), embedding_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()

    x1 = torch.cat([torch.zeros(8, 2), torch.zeros(8, 2)])
    x2 = torch.cat([torch.zeros(8, 2), 4.0 * torch.ones(8, 2)])
    labels = torch.cat([torch.ones(8), torch.zeros(8)])

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(x1, x2), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


# §6 scorer property (read-only by design)


def test_scorer_is_read_only(bce_model: BCESiameseNetwork) -> None:
    """Assigning to ``scorer`` raises instead of silently registering a module."""
    original_scorer = bce_model.scorer
    with pytest.raises(AttributeError):
        bce_model.scorer = torch.nn.Linear(EMBEDDING_DIM, 1)  # type: ignore[misc]
    assert "scorer" not in bce_model._modules
    assert bce_model.scorer is original_scorer


# §7 Oxford Flowers integration: real data through the stub backbone


def test_embed_on_flower_images_raises(
    bce_model: BCESiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Real images are no exception: this network exposes no embedding."""
    with pytest.raises(NotImplementedError):
        bce_model.embed(flower_subset[0][0])


def test_identical_flower_scores_sigmoid_of_bias(
    bce_model: BCESiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """A flower compared with itself scores the learned operating point."""
    image = flower_subset[0][0]
    score = bce_model.similarity_score(image, image.copy())
    assert isinstance(bce_model.scorer, torch.nn.Linear)
    bias = float(bce_model.scorer.bias.detach().cpu())
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(_sigmoid(bias), abs=1e-5)


def test_similarity_matrix_on_flowers(
    bce_model: BCESiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Batches of flowers produce an ``(N, M)`` matrix of probabilities."""
    batch_a = [flower_subset[index][0] for index in (0, 1)]
    batch_b = [flower_subset[index][0] for index in (2, 3, 4)]
    score = bce_model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)
    assert np.all(score > 0.0)
    assert np.all(score < 1.0)


def test_flower_similarity_is_symmetric(
    bce_model: BCESiameseNetwork, flower_subset: OxfordFlowerDataset
) -> None:
    """Swapping two real flower images transposes the probability matrix."""
    batch_a = [flower_subset[index][0] for index in (0, 1)]
    batch_b = [flower_subset[index][0] for index in (2, 3, 4)]
    forward_score = bce_model.similarity_score(batch_a, batch_b)
    backward_score = bce_model.similarity_score(batch_b, batch_a)
    assert np.allclose(forward_score, backward_score.T, atol=1e-6)
