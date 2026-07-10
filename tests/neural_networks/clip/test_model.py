"""Fast unit tests for pyvisim's CLIP image-tower implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pyvisim.neural_networks.clip import VisionConfig
from pyvisim.neural_networks.clip._model import (
    QuickGELU,
    build_vision_model,
    load_vision_weights,
)

#: Tiny ViT: 2 blocks of width 16 (2 heads), 2x2 patch grid, 8-dim output.
TINY_VIT = VisionConfig(
    embed_dim=8, image_size=32, width=16, layers=2, patch_size=16, head_width=8
)

#: Tiny ResNet: one bottleneck per stage, stem width 8, 8-dim output.
TINY_RESNET = VisionConfig(embed_dim=8, image_size=32, width=8, layers=(1, 1, 1, 1))


def _checkpoint_file(model: torch.nn.Module, path: Path, **extra: torch.Tensor) -> Path:
    """Save a model's weights as an open_clip-style safetensors checkpoint."""
    state = {f"visual.{key}": value for key, value in model.state_dict().items()}
    state.update(extra)
    save_file(state, str(path))
    return path


# §1 architecture


@pytest.mark.parametrize("config", [TINY_VIT, TINY_RESNET], ids=["vit", "resnet"])
def test_forward_maps_images_to_embeddings(config: VisionConfig) -> None:
    """Both towers map an image batch to ``(N, embed_dim)`` embeddings."""
    model = build_vision_model(config, quick_gelu=False).eval()
    batch = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (2, config.embed_dim)
    assert output.dtype == torch.float32


def test_vit_state_dict_uses_open_clip_naming() -> None:
    """The ViT parameter names match open_clip's ``visual.*`` entries."""
    keys = set(build_vision_model(TINY_VIT, quick_gelu=False).state_dict())
    assert {
        "class_embedding",
        "positional_embedding",
        "proj",
        "conv1.weight",
        "ln_pre.weight",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.out_proj.weight",
        "transformer.resblocks.1.mlp.c_fc.weight",
        "transformer.resblocks.1.mlp.c_proj.bias",
        "ln_post.weight",
    } <= keys


def test_resnet_state_dict_uses_open_clip_naming() -> None:
    """The ResNet parameter names match open_clip's ``visual.*`` entries."""
    keys = set(build_vision_model(TINY_RESNET, quick_gelu=False).state_dict())
    assert {
        "conv1.weight",
        "bn3.running_var",
        "layer1.0.conv1.weight",
        "layer1.0.downsample.0.weight",
        "layer2.0.downsample.1.bias",
        "attnpool.positional_embedding",
        "attnpool.q_proj.weight",
        "attnpool.c_proj.bias",
    } <= keys


def test_quick_gelu_matches_its_published_formula() -> None:
    """QuickGELU computes ``x * sigmoid(1.702 * x)``."""
    x = torch.linspace(-3.0, 3.0, 13)
    assert torch.equal(QuickGELU()(x), x * torch.sigmoid(1.702 * x))


def test_quick_gelu_changes_the_vit_output() -> None:
    """The same ViT weights produce different embeddings per activation."""
    gelu_model = build_vision_model(TINY_VIT, quick_gelu=False).eval()
    quick_model = build_vision_model(TINY_VIT, quick_gelu=True).eval()
    quick_model.load_state_dict(gelu_model.state_dict())
    batch = torch.randn(1, 3, TINY_VIT.image_size, TINY_VIT.image_size)
    with torch.no_grad():
        assert not torch.allclose(gelu_model(batch), quick_model(batch))


def test_vit_config_without_patch_size_raises() -> None:
    """Building a ViT from a config missing ``patch_size`` raises."""
    config = VisionConfig(embed_dim=8, image_size=32, width=16, layers=2)
    with pytest.raises(ValueError, match="patch_size"):
        build_vision_model(config, quick_gelu=False)


# §2 weight loading


@pytest.mark.parametrize("config", [TINY_VIT, TINY_RESNET], ids=["vit", "resnet"])
def test_load_vision_weights_restores_the_saved_model(
    config: VisionConfig, tmp_path: Path
) -> None:
    """Weights written with the ``visual.`` prefix load back exactly."""
    source = build_vision_model(config, quick_gelu=False).eval()
    path = _checkpoint_file(
        source, tmp_path / "model.safetensors", text_projection=torch.zeros(2)
    )
    target = build_vision_model(config, quick_gelu=False).eval()
    load_vision_weights(target, path)
    batch = torch.randn(1, 3, config.image_size, config.image_size)
    with torch.no_grad():
        assert torch.equal(source(batch), target(batch))


def test_load_vision_weights_casts_half_precision_to_float32(
    tmp_path: Path,
) -> None:
    """A half-precision checkpoint is loaded as ``float32`` weights."""
    source = build_vision_model(TINY_VIT, quick_gelu=False).half()
    path = _checkpoint_file(source, tmp_path / "fp16.safetensors")
    target = build_vision_model(TINY_VIT, quick_gelu=False)
    load_vision_weights(target, path)
    assert all(value.dtype == torch.float32 for value in target.state_dict().values())


def test_load_vision_weights_tolerates_missing_batchnorm_counters(
    tmp_path: Path,
) -> None:
    """BatchNorm ``num_batches_tracked`` entries may be absent."""
    source = build_vision_model(TINY_RESNET, quick_gelu=False)
    state = {
        f"visual.{key}": value
        for key, value in source.state_dict().items()
        if not key.endswith("num_batches_tracked")
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    target = build_vision_model(TINY_RESNET, quick_gelu=False)
    load_vision_weights(target, tmp_path / "model.safetensors")


def test_load_vision_weights_without_image_tower_raises(tmp_path: Path) -> None:
    """A checkpoint without ``visual.*`` entries raises a ``ValueError``."""
    save_file({"text_projection": torch.zeros(2)}, str(tmp_path / "text.safetensors"))
    with pytest.raises(ValueError, match="no image-tower weights"):
        load_vision_weights(
            build_vision_model(TINY_VIT, quick_gelu=False),
            tmp_path / "text.safetensors",
        )


def test_load_vision_weights_architecture_mismatch_raises(tmp_path: Path) -> None:
    """Loading a checkpoint of a different architecture raises."""
    source = build_vision_model(TINY_RESNET, quick_gelu=False)
    path = _checkpoint_file(source, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="does not match the model architecture"):
        load_vision_weights(build_vision_model(TINY_VIT, quick_gelu=False), path)
