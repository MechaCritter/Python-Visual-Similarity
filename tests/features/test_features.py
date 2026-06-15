"""Tests for the feature extractors in :mod:`pyvisim.features`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyvisim.features import SIFT, DeepConvFeature, Lambda, RootSIFT

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Image fixtures of varying sizes; extractors must honour the shape contract
#: regardless of input size (the requirement only exercises extractors here).
VARYING_SIZE_FIXTURES = ["small_image", "large_image", "non_square_image", "rgb_image"]


def _tiny_conv_model() -> nn.Module:
    """Build a tiny two-conv-layer network to avoid downloading VGG16.

    :returns: a ``torch.nn.Sequential`` with two ``Conv2d`` layers (6 then 10
        output channels) separated by a ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(3, 6, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 10, 3, padding=1),
    )


# ---------------------------------------------------------------------------
# SIFT
# ---------------------------------------------------------------------------


def test_sift_output_dim() -> None:
    """SIFT descriptors are 128-dimensional."""
    assert SIFT().output_dim == 128


def test_sift_extracts_descriptors(checkerboard_image: ImageObj) -> None:
    """A corner-rich image yields a non-empty ``(N, 128)`` float32 matrix."""
    out = SIFT()(checkerboard_image.array)
    assert out.ndim == 2
    assert out.shape[1] == 128
    assert out.shape[0] > 0
    assert out.dtype == np.float32


def test_sift_featureless_returns_empty(solid_image: ImageObj) -> None:
    """A featureless image yields an empty ``(0, 128)`` array."""
    assert SIFT()(solid_image.array).shape == (0, 128)


def test_sift_stripes_returns_empty(stripes_image: ImageObj) -> None:
    """A stripe pattern has no SIFT corners, yielding an empty array."""
    assert SIFT()(stripes_image.array).shape == (0, 128)


def test_sift_tiny_returns_empty(tiny_image: ImageObj) -> None:
    """An 8x8 image is too small for keypoints, yielding an empty array."""
    assert SIFT()(tiny_image.array).shape == (0, 128)


def test_sift_accepts_tensor(checkerboard_image: ImageObj) -> None:
    """A grayscale torch tensor is accepted and yields the ``(N, 128)`` contract."""
    tensor = torch.from_numpy(checkerboard_image.array)
    out = SIFT()(tensor)
    assert out.ndim == 2
    assert out.shape[1] == 128
    assert out.shape[0] > 0


def test_sift_repr() -> None:
    """``repr`` reports the extractor name and its output dimension."""
    assert repr(SIFT()) == "SIFT(output_dim=128)"


@pytest.mark.parametrize("image_fixture", VARYING_SIZE_FIXTURES)
def test_sift_varying_sizes(request: pytest.FixtureRequest, image_fixture: str) -> None:
    """SIFT honours the ``(N, 128)`` shape contract across input sizes."""
    image = request.getfixturevalue(image_fixture).array
    out = SIFT()(image)
    assert out.ndim == 2
    assert out.shape[1] == 128


# RootSIFT (default extractor)


def test_rootsift_output_dim() -> None:
    """RootSIFT descriptors are 128-dimensional."""
    assert RootSIFT().output_dim == 128


def test_rootsift_extracts_descriptors(checkerboard_image: ImageObj) -> None:
    """A corner-rich image yields a non-empty ``(N, 128)`` float32 matrix."""
    out = RootSIFT()(checkerboard_image.array)
    assert out.ndim == 2
    assert out.shape[1] == 128
    assert out.shape[0] > 0
    assert out.dtype == np.float32


def test_rootsift_values_non_negative(checkerboard_image: ImageObj) -> None:
    """RootSIFT values are non-negative (square root of a normalized histogram)."""
    out = RootSIFT()(checkerboard_image.array)
    assert np.all(out >= 0)


def test_rootsift_rows_l2_normalized(checkerboard_image: ImageObj) -> None:
    """Each RootSIFT descriptor row has unit L2 norm."""
    out = RootSIFT()(checkerboard_image.array)
    norms = np.linalg.norm(out, axis=1)
    assert norms == pytest.approx(np.ones_like(norms), rel=1e-3)


def test_rootsift_featureless_returns_empty(solid_image: ImageObj) -> None:
    """A featureless image yields an empty ``(0, 128)`` array."""
    assert RootSIFT()(solid_image.array).shape == (0, 128)


def test_rootsift_low_contrast_returns_empty() -> None:
    """A low-contrast checkerboard (values 120/136) yields no descriptors."""
    rows, cols = np.indices((256, 256)) // 16
    pattern = (rows + cols) % 2
    image = np.where(pattern == 0, 120, 136).astype(np.uint8)
    assert RootSIFT()(image).shape == (0, 128)


def test_rootsift_accepts_tensor(checkerboard_image: ImageObj) -> None:
    """A grayscale torch tensor is accepted and yields the ``(N, 128)`` contract."""
    tensor = torch.from_numpy(checkerboard_image.array)
    out = RootSIFT()(tensor)
    assert out.ndim == 2
    assert out.shape[1] == 128
    assert out.shape[0] > 0


def test_rootsift_repr() -> None:
    """``repr`` reports the extractor name and its output dimension."""
    assert repr(RootSIFT()) == "RootSIFT(output_dim=128)"


@pytest.mark.parametrize("image_fixture", VARYING_SIZE_FIXTURES)
def test_rootsift_varying_sizes(
    request: pytest.FixtureRequest, image_fixture: str
) -> None:
    """RootSIFT honours the ``(N, 128)`` shape contract across input sizes."""
    image = request.getfixturevalue(image_fixture).array
    out = RootSIFT()(image)
    assert out.ndim == 2
    assert out.shape[1] == 128


# Lambda


def test_lambda_non_callable_raises() -> None:
    """Constructing ``Lambda`` with a non-callable raises ``ValueError``."""
    with pytest.raises(ValueError, match="must be a callable"):
        Lambda(123, output_dim=4)  # type: ignore[arg-type]


def test_lambda_output_dim() -> None:
    """``output_dim`` reflects the constructor argument."""
    extractor = Lambda(lambda image: np.ones((5, 4), np.float32), output_dim=4)
    assert extractor.output_dim == 4


def test_lambda_happy_path(checkerboard_image: ImageObj) -> None:
    """A well-behaved function passes through the shape validation."""
    extractor = Lambda(lambda image: np.ones((5, 4), np.float32), output_dim=4)
    out = extractor(checkerboard_image.array)
    assert out.shape == (5, 4)
    assert out.dtype == np.float32


def test_lambda_wrong_dim_raises(checkerboard_image: ImageObj) -> None:
    """A function whose output width mismatches ``output_dim`` raises ``ValueError``."""
    extractor = Lambda(lambda image: np.ones((5, 3), np.float32), output_dim=4)
    with pytest.raises(ValueError, match=r"Expected feat_vecs.shape\[1\] == 4"):
        extractor(checkerboard_image.array)


def test_lambda_non_2d_raises(checkerboard_image: ImageObj) -> None:
    """A function returning a non-2D array raises ``ValueError``."""
    extractor = Lambda(lambda image: np.ones((4,), np.float32), output_dim=4)
    with pytest.raises(ValueError, match="must be 2D"):
        extractor(checkerboard_image.array)


def test_lambda_non_ndarray_raises(checkerboard_image: ImageObj) -> None:
    """A function returning a non-array raises ``ValueError``."""
    extractor = Lambda(lambda image: [[1, 2, 3, 4]], output_dim=4)
    with pytest.raises(ValueError, match="Expected output to be a NumPy array"):
        extractor(checkerboard_image.array)


def test_lambda_none_returns_empty(checkerboard_image: ImageObj) -> None:
    """A function returning ``None`` yields an empty ``(0, output_dim)`` array."""
    extractor = Lambda(lambda image: None, output_dim=4)
    assert extractor(checkerboard_image.array).shape == (0, 4)


def test_lambda_accepts_tensor() -> None:
    """A torch tensor is normalized and passed through to the function."""
    extractor = Lambda(lambda image: np.ones((5, 4), np.float32), output_dim=4)
    out = extractor(torch.zeros(8, 8))
    assert out.shape == (5, 4)


# DeepConvFeature


def test_deepconv_output_dim_with_spatial() -> None:
    """With spatial encoding, ``output_dim`` is channels + 2."""
    extractor = DeepConvFeature(
        _tiny_conv_model(), layer_index=-1, spatial_encoding=True, device="cpu"
    )
    assert extractor.output_dim == 12


def test_deepconv_output_dim_no_spatial() -> None:
    """Without spatial encoding, ``output_dim`` is the channel count."""
    extractor = DeepConvFeature(
        _tiny_conv_model(), layer_index=-1, spatial_encoding=False, device="cpu"
    )
    assert extractor.output_dim == 10


def test_deepconv_call_shape_with_spatial(rgb_image: ImageObj) -> None:
    """The call flattens the feature map to ``(Hf*Wf, output_dim)`` float32."""
    extractor = DeepConvFeature(
        _tiny_conv_model(), layer_index=-1, spatial_encoding=True, device="cpu"
    )
    out = extractor(rgb_image.array)
    assert out.ndim == 2
    assert out.shape[1] == 12
    assert out.shape[0] == 224 * 224
    assert out.dtype == np.float32


def test_deepconv_call_shape_no_spatial(rgb_image: ImageObj) -> None:
    """Without spatial encoding the flattened map has ``output_dim`` columns."""
    extractor = DeepConvFeature(
        _tiny_conv_model(), layer_index=-1, spatial_encoding=False, device="cpu"
    )
    out = extractor(rgb_image.array)
    assert out.shape == (224 * 224, 10)


def test_deepconv_list_conv_layers() -> None:
    """``list_conv_layers`` enumerates the conv layers as ``(idx, name, module)``."""
    extractor = DeepConvFeature(_tiny_conv_model(), device="cpu")
    layers = extractor.list_conv_layers()
    assert len(layers) == 2
    for index, name, module in layers:
        assert isinstance(index, int)
        assert isinstance(name, str)
        assert isinstance(module, nn.Conv2d)


def test_deepconv_no_conv_layers_raises() -> None:
    """A model with no conv layers raises ``ValueError``."""
    with pytest.raises(ValueError, match="No convolutional layers found"):
        DeepConvFeature(nn.Sequential(nn.ReLU()), device="cpu")


def test_deepconv_layer_index_out_of_range_raises() -> None:
    """A ``layer_index`` beyond the available conv layers raises ``IndexError``."""
    with pytest.raises(IndexError, match="has only 2 convolutional layers"):
        DeepConvFeature(_tiny_conv_model(), layer_index=5, device="cpu")


def test_deepconv_bad_submodule_raises() -> None:
    """An unknown ``target_submodule`` raises ``AttributeError``."""
    with pytest.raises(AttributeError):
        DeepConvFeature(
            _tiny_conv_model(), target_submodule="does_not_exist", device="cpu"
        )


def test_deepconv_non_module_model_raises() -> None:
    """A model that is not a ``torch.nn.Module`` raises ``TypeError``."""
    with pytest.raises(TypeError):
        DeepConvFeature(model="not a module", device="cpu")  # type: ignore[arg-type]


def test_deepconv_accepts_tensor() -> None:
    """A ``CHW`` torch tensor is accepted when its layout is described via ``dims``."""
    extractor = DeepConvFeature(_tiny_conv_model(), device="cpu")
    out = extractor(torch.zeros(3, 64, 64), dims="CHW")
    assert out.ndim == 2


@pytest.mark.slow
def test_deepconv_default_vgg16() -> None:
    """The default VGG16 last conv layer gives ``output_dim == 514`` (512 + 2)."""
    extractor = DeepConvFeature(device="cpu")
    assert extractor.output_dim == 514
