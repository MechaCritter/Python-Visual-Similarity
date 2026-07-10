"""Fast unit tests for the CLIP variant/checkpoint registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyvisim.neural_networks.clip import (
    MODEL_CONFIGS,
    PRETRAINED,
    available_pretrained,
    available_variants,
    fetch_checkpoint,
    get_checkpoint_spec,
    get_model_config,
)
from pyvisim.neural_networks.clip import _registry as registry_module

# §1 registry integrity


def test_every_variant_has_a_config_and_pretrained_tags() -> None:
    """Model configs and pretrained tags cover exactly the same variants."""
    assert set(MODEL_CONFIGS) == set(PRETRAINED)
    assert available_variants() == tuple(MODEL_CONFIGS)
    assert all(PRETRAINED[variant] for variant in PRETRAINED)


def test_registry_mirrors_open_clips_supported_subset() -> None:
    """The registry holds the 67 (variant, tag) pairs of open_clip's registry
    that use a standard vision tower and publish safetensors weights."""
    assert len(MODEL_CONFIGS) == 30
    assert sum(len(tags) for tags in PRETRAINED.values()) == 67


def test_every_spec_is_well_formed() -> None:
    """Every checkpoint points at a Hub repo and a safetensors file."""
    for tags in PRETRAINED.values():
        for spec in tags.values():
            assert spec.repo_id.count("/") == 1
            assert spec.filename.endswith(".safetensors")
            assert spec.resize_mode in ("shortest", "squash")
            assert len(spec.mean) == len(spec.std) == 3
            assert spec.url == f"https://huggingface.co/{spec.repo_id}"


def test_openai_checkpoints_use_quick_gelu() -> None:
    """Every original OpenAI checkpoint is flagged as QuickGELU-trained."""
    openai_specs = [tags["openai"] for tags in PRETRAINED.values() if "openai" in tags]
    assert openai_specs
    assert all(spec.quick_gelu for spec in openai_specs)


def test_quickgelu_variants_mirror_their_base_variant() -> None:
    """A ``-quickgelu`` variant shares its base variant's architecture."""
    for variant in MODEL_CONFIGS:
        if variant.endswith("-quickgelu"):
            base = variant.removesuffix("-quickgelu")
            assert MODEL_CONFIGS[variant] == MODEL_CONFIGS[base]


def test_configs_match_the_published_model_configurations() -> None:
    """Spot-check dimensions against the official open_clip model configs."""
    assert get_model_config("ViT-B-32").embed_dim == 512
    assert get_model_config("ViT-B-32").heads == 12
    assert get_model_config("ViT-B-32").mlp_width == 3072
    assert get_model_config("ViT-L-14").embed_dim == 768
    assert get_model_config("ViT-L-14-336").image_size == 336
    assert get_model_config("ViT-H-14").heads == 16
    assert get_model_config("ViT-g-14").mlp_width == 6144
    assert get_model_config("ViT-bigG-14").mlp_width == 8192
    assert get_model_config("RN50").embed_dim == 1024
    assert get_model_config("RN50").heads == 32
    assert get_model_config("RN50x64").image_size == 448


# §2 lookups and aliases


def test_openai_style_variant_names_are_aliases() -> None:
    """OpenAI-style spellings map onto the open_clip-style names."""
    assert get_model_config("ViT-B/32") == get_model_config("ViT-B-32")
    assert get_checkpoint_spec("ViT-B/16", "openai") == get_checkpoint_spec(
        "ViT-B-16", "openai"
    )
    assert available_pretrained("ViT-L/14") == available_pretrained("ViT-L-14")


def test_get_model_config_unknown_variant_raises() -> None:
    """An unknown variant name raises a ``ValueError`` naming the variant."""
    with pytest.raises(ValueError, match="ViT-B-64"):
        get_model_config("ViT-B/64")


def test_get_checkpoint_spec_unknown_variant_raises() -> None:
    """An unknown variant raises before any tag lookup."""
    with pytest.raises(ValueError, match="Unsupported CLIP variant"):
        get_checkpoint_spec("ViT-B-64", "openai")


def test_get_checkpoint_spec_unknown_tag_lists_available_tags() -> None:
    """An unknown tag raises a ``ValueError`` listing the supported tags."""
    with pytest.raises(ValueError, match="laion2b_s34b_b79k"):
        get_checkpoint_spec("ViT-B-32", "no_such_tag")


def test_available_pretrained_unknown_variant_raises() -> None:
    """An unknown variant name raises a ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported CLIP variant"):
        available_pretrained("ConvNeXt-B")


# §3 checkpoint fetching


def test_fetch_checkpoint_downloads_the_specs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec's repository and file name are handed to huggingface_hub."""
    recorded: dict[str, object] = {}

    def fake_download(
        repo_id: str, filename: str, cache_dir: str | Path | None = None
    ) -> str:
        recorded.update(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return str(tmp_path / filename)

    monkeypatch.setattr(registry_module, "hf_hub_download", fake_download)
    path = fetch_checkpoint("ViT-B-32", "laion2b_s34b_b79k", cache_dir=tmp_path)
    assert path == tmp_path / "open_clip_model.safetensors"
    assert recorded == {
        "repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "filename": "open_clip_model.safetensors",
        "cache_dir": tmp_path,
    }


def test_fetch_checkpoint_unknown_variant_raises_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported variant fails fast, without touching the network."""
    monkeypatch.setattr(
        registry_module,
        "hf_hub_download",
        lambda *args, **kwargs: pytest.fail("must not download"),
    )
    with pytest.raises(ValueError, match="Unsupported CLIP variant"):
        fetch_checkpoint("ViT-B-64", "openai")
