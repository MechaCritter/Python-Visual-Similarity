"""Unit tests for :class:`pyvisim.neural_networks.SiameseNeuralNetwork`.

The model resolves backbone *names* to pretrained networks, so these tests
install tiny deterministic stub backbones (see :mod:`._stubs`) instead: the
embedding math then becomes exactly hand-computable and no weights are ever
downloaded. Integration with the real ResNet-18 is covered by the Oxford
Flowers tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import transforms

from pyvisim._errors import InvalidImageError
from pyvisim.neural_networks import SiameseNeuralNetwork
from pyvisim.typing import FloatNumpyArray

from ._stubs import (
    FlattenBackbone,
    MeanBackbone,
    build_stub_model,
    install_head,
    make_identity_head,
    make_random_rgb_image,
    make_solid_rgb_image,
)

# §1 construction and backbone resolution


def test_unknown_backbone_name_raises() -> None:
    """An unrecognised backbone name raises ``ValueError`` at construction."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        SiameseNeuralNetwork(backbone="alexnet")


def test_get_backbone_rejects_unknown_name() -> None:
    """The backbone resolver raises ``ValueError`` for unknown names."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        SiameseNeuralNetwork._get_backbone(
            "definitely-not-a-backbone", pretrained=False
        )


def test_non_pretrained_backbone_builds_without_weights() -> None:
    """``pretrained_backbone=False`` builds a real ResNet-18 without weights."""
    model = SiameseNeuralNetwork(embedding_dim=8, pretrained_backbone=False)
    embeddings = model.encode(make_random_rgb_image(seed=1))
    assert embeddings.shape == (1, 8)


def test_default_embedding_dim_is_128() -> None:
    """The projection head defaults to a 128-dimensional embedding space."""
    model = build_stub_model(MeanBackbone())
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.out_features == 128


def test_head_input_matches_backbone_output_dim() -> None:
    """The head's ``in_features`` equals the backbone's ``output_dim``."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4)
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.in_features == MeanBackbone.output_dim


def test_device_attribute_is_torch_device() -> None:
    """The ``device`` string is normalized to a ``torch.device``."""
    model = build_stub_model(MeanBackbone(), device="cpu")
    assert model.device == torch.device("cpu")


# §2 default ImageNet transform


def test_imagenet_transform_for_resnet18() -> None:
    """The resnet18 default transform is the documented ImageNet pipeline.

    Resize to 256, center-crop to 224, convert to tensor and normalize with
    the ImageNet channel statistics.
    """
    transform = SiameseNeuralNetwork._get_imagenet_transform("resnet18")
    steps = transform.transforms
    assert [type(step) for step in steps] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert steps[0].size == 256
    assert steps[1].size == (224, 224)
    assert steps[3].mean == [0.485, 0.456, 0.406]
    assert steps[3].std == [0.229, 0.224, 0.225]


def test_imagenet_transform_unknown_backbone_raises() -> None:
    """Requesting the default transform for an unknown backbone raises."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        SiameseNeuralNetwork._get_imagenet_transform("alexnet")


# §3 forward: mathematical correctness on trivial examples


def test_forward_l2_normalizes_embeddings() -> None:
    """With identity backbone and head, ``forward`` is plain L2 normalization.

    The input row ``[3, 4]`` has norm 5, so the embedding must be
    ``[0.6, 0.8]``.
    """
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2)
    install_head(model, make_identity_head(2))
    embedding = model(torch.tensor([[3.0, 4.0], [0.0, 5.0]]))
    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    assert torch.allclose(embedding, expected, atol=1e-6)


def test_forward_output_has_unit_norm() -> None:
    """Every embedding row of a random batch has unit L2 norm."""
    torch.manual_seed(0)
    model = build_stub_model(FlattenBackbone(8), embedding_dim=5)
    embeddings = model(torch.randn(16, 8))
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones(16), atol=1e-5)


def test_forward_applies_head_before_normalizing() -> None:
    """With head weights ``diag(1, 2)``, input ``[1, 1]`` maps to ``[1, 2]/sqrt(5)``."""
    model = build_stub_model(FlattenBackbone(2), embedding_dim=2)
    head = torch.nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        head.bias.zero_()
    install_head(model, head)
    embedding = model(torch.tensor([[1.0, 1.0]]))
    expected = torch.tensor([[1.0, 2.0]]) / math.sqrt(5.0)
    assert torch.allclose(embedding, expected, atol=1e-6)


# §4 encode: shapes, layouts and value ranges


@pytest.fixture
def model() -> SiameseNeuralNetwork:
    """A small deterministic model for image-level tests.

    :return: Model with a :class:`MeanBackbone` and a 4-dim embedding space.
    """
    torch.manual_seed(0)
    return build_stub_model(MeanBackbone(), embedding_dim=4)


def test_encode_single_image_shape_and_norm(model: SiameseNeuralNetwork) -> None:
    """One HWC image encodes to a single unit-norm float32 row."""
    embeddings = model.encode(make_random_rgb_image(seed=1))
    assert embeddings.shape == (1, 4)
    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_encode_list_of_images(model: SiameseNeuralNetwork) -> None:
    """A list of N images encodes to an ``(N, embedding_dim)`` batch."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2, 3)]
    embeddings = model.encode(images)
    assert embeddings.shape == (3, 4)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0] * 3, abs=1e-5)


def test_encode_batch_matches_single_encoding(model: SiameseNeuralNetwork) -> None:
    """Batched encoding equals encoding each image individually."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2, 3)]
    batched = model.encode(images)
    single = model.encode(images[0])
    assert np.allclose(batched[0], single[0], atol=1e-6)


def test_encode_bhwc_array_matches_list(model: SiameseNeuralNetwork) -> None:
    """A stacked BHWC array yields the same embeddings as a list of images."""
    images = [make_random_rgb_image(seed=s) for s in (1, 2)]
    from_list = model.encode(images)
    from_batch = model.encode(np.stack(images), dims="BHWC")
    assert np.array_equal(from_list, from_batch)


def test_encode_chw_matches_hwc(model: SiameseNeuralNetwork) -> None:
    """The same image in CHW layout yields the same embedding as in HWC."""
    image = make_random_rgb_image(seed=7)
    from_hwc = model.encode(image)
    from_chw = model.encode(image.transpose(2, 0, 1), dims="CHW")
    assert np.array_equal(from_hwc, from_chw)


def test_encode_unit_value_range_matches_uint8(model: SiameseNeuralNetwork) -> None:
    """A float image in [0, 1] encodes identically to its uint8 counterpart."""
    checker = np.zeros((8, 8, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    from_uint8 = model.encode(checker)
    from_float = model.encode(
        checker.astype(np.float64) / 255.0, value_range=(0.0, 1.0)
    )
    assert np.array_equal(from_uint8, from_float)


def test_encode_grayscale_image(model: SiameseNeuralNetwork) -> None:
    """A single-channel image is promoted to RGB and encoded normally."""
    gray = np.full((16, 16), 100, dtype=np.uint8)
    embeddings = model.encode(gray, dims="HW")
    assert embeddings.shape == (1, 4)


def test_encode_empty_input_raises(model: SiameseNeuralNetwork) -> None:
    """An empty image collection raises ``ValueError``."""
    with pytest.raises(ValueError, match="at least one image"):
        model.encode([])


def test_encode_rejects_string_input(model: SiameseNeuralNetwork) -> None:
    """A string is not an image and raises ``InvalidImageError``."""
    with pytest.raises(InvalidImageError):
        model.encode("not-an-image.jpg")


def test_encode_is_deterministic(model: SiameseNeuralNetwork) -> None:
    """Encoding the same image twice yields identical embeddings."""
    image = make_random_rgb_image(seed=5)
    assert np.array_equal(model.encode(image), model.encode(image))


def test_encode_restores_training_mode(model: SiameseNeuralNetwork) -> None:
    """``encode`` runs in eval mode but restores the previous training state."""
    model.train()
    model.encode(make_solid_rgb_image(value=10))
    assert model.training is True
    model.eval()
    model.encode(make_solid_rgb_image(value=10))
    assert model.training is False


# §5 encode: mathematical correctness through the full public path


def test_encode_of_uniform_image_is_uniform_unit_vector() -> None:
    """A constant image maps to the constant unit vector ``1/sqrt(D)``.

    With a plain ``ToTensor`` transform, a 2x2 uniform image becomes 12 equal
    feature values; the identity head then L2-normalizes them to
    ``1/sqrt(12)`` each.
    """
    model = build_stub_model(
        FlattenBackbone(12),
        embedding_dim=12,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(12))
    embeddings = model.encode(make_solid_rgb_image(value=51, size=2))
    assert embeddings.shape == (1, 12)
    assert np.allclose(embeddings, 1.0 / math.sqrt(12.0), atol=1e-6)


def test_encode_preserves_pixel_ratios() -> None:
    """Pixel intensities 30 and 40 embed as the 3:4 pattern of unit norm.

    A 1x2 image with pixel values 30 and 40 flattens channel-major to
    ``[30, 40, 30, 40, 30, 40]``; normalization cancels the ``/255`` scale,
    leaving exactly ``[3, 4, 3, 4, 3, 4] / (5 * sqrt(3))``.
    """
    model = build_stub_model(
        FlattenBackbone(6),
        embedding_dim=6,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    install_head(model, make_identity_head(6))
    image = np.array([[[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    embeddings = model.encode(image)
    expected = np.array([3.0, 4.0] * 3) / (5.0 * math.sqrt(3.0))
    assert np.allclose(embeddings[0], expected, atol=1e-6)


# §6 similarity_score


def test_similarity_of_identical_images_is_one(model: SiameseNeuralNetwork) -> None:
    """Cosine similarity of an image with itself is 1."""
    image = make_random_rgb_image(seed=11)
    score = model.similarity_score(image, image.copy())
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_similarity_matrix_shape(model: SiameseNeuralNetwork) -> None:
    """Two batches of sizes N and M produce an ``(N, M)`` similarity matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = model.similarity_score(batch_a, batch_b)
    assert score.shape == (2, 3)


def test_similarity_scores_bounded_by_one(model: SiameseNeuralNetwork) -> None:
    """Cosine similarities of unit embeddings lie in ``[-1, 1]``."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    score = model.similarity_score(batch_a, batch_b)
    assert np.all(score >= -1.0 - 1e-6)
    assert np.all(score <= 1.0 + 1e-6)


def test_similarity_equals_embedding_dot_product(
    model: SiameseNeuralNetwork,
) -> None:
    """On L2-normalized embeddings, cosine similarity is the dot product."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4)]
    score = model.similarity_score(batch_a, batch_b)
    expected = model.encode(batch_a) @ model.encode(batch_b).T
    assert np.allclose(score, expected, atol=1e-5)


def test_similarity_is_symmetric(model: SiameseNeuralNetwork) -> None:
    """Swapping the inputs transposes the cosine similarity matrix."""
    batch_a = [make_random_rgb_image(seed=s) for s in (1, 2)]
    batch_b = [make_random_rgb_image(seed=s) for s in (3, 4, 5)]
    forward_score = model.similarity_score(batch_a, batch_b)
    backward_score = model.similarity_score(batch_b, batch_a)
    assert np.allclose(forward_score, backward_score.T, atol=1e-6)


def test_custom_similarity_func_is_used() -> None:
    """A user-provided similarity function replaces the cosine default."""

    def constant_score(x: FloatNumpyArray, y: FloatNumpyArray) -> FloatNumpyArray:
        return np.full((x.shape[0], y.shape[0]), 0.5)

    model = build_stub_model(
        MeanBackbone(), embedding_dim=4, similarity_func=constant_score
    )
    score = model.similarity_score(
        make_random_rgb_image(seed=1), make_random_rgb_image(seed=2)
    )
    assert np.array_equal(score, np.full((1, 1), 0.5))


# §7 head and backbone properties (read-only by design)


def test_head_property_returns_current_head(model: SiameseNeuralNetwork) -> None:
    """The property exposes the projection head module."""
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.out_features == 4


def test_backbone_property_returns_current_backbone(
    model: SiameseNeuralNetwork,
) -> None:
    """The property exposes the feature-extraction backbone module."""
    assert isinstance(model.backbone, MeanBackbone)


def test_head_is_read_only(model: SiameseNeuralNetwork) -> None:
    """Assigning to ``head`` raises instead of silently registering a module.

    ``torch.nn.Module.__setattr__`` would otherwise register the value as an
    orphan ``head`` submodule whose parameters get optimized but never used.
    """
    original_head = model.head
    with pytest.raises(AttributeError):
        model.head = torch.nn.Linear(3, 8)  # type: ignore[misc]
    assert "head" not in model._modules
    assert model.head is original_head


def test_backbone_is_read_only(model: SiameseNeuralNetwork) -> None:
    """Assigning to ``backbone`` raises instead of silently registering a module."""
    original_backbone = model.backbone
    with pytest.raises(AttributeError):
        model.backbone = MeanBackbone()  # type: ignore[misc]
    assert "backbone" not in model._modules
    assert model.backbone is original_backbone


# §8 device handling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_encode_on_cuda_returns_cpu_numpy() -> None:
    """A CUDA model still returns NumPy embeddings on the host."""
    model = build_stub_model(MeanBackbone(), embedding_dim=4, device="cuda")
    assert model.device.type == "cuda"
    embeddings = model.encode(make_random_rgb_image(seed=1))
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 4)
