"""A tiny, test-only CLIP variant shared by the ``clip`` test modules.

The real checkpoints weigh hundreds of megabytes, so the tests register a
tiny variant on top of the real registry and write its checkpoint to disk as
a real safetensors file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pyvisim.neural_networks.clip import CheckpointSpec, VisionConfig
from pyvisim.neural_networks.clip import _registry as registry_module
from pyvisim.neural_networks.clip import clip_embedder as clip_embedder_module
from pyvisim.neural_networks.clip._model import build_vision_model

#: Name, tag and architecture of the tiny test-only variant.
VARIANT = "tiny-ViT"
TAG = "test"
CONFIG = VisionConfig(
    embed_dim=8, image_size=32, width=16, layers=2, patch_size=16, head_width=8
)


def register_tiny_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """
    Register the tiny variant and write its checkpoint into ``tmp_path``.

    :param monkeypatch: The patcher whose scope the registration lives in.
    :param tmp_path: Directory the tiny checkpoint is written to.
    :return: The name of the registered variant.
    """
    checkpoint = tmp_path / "open_clip_model.safetensors"
    torch.manual_seed(0)
    model = build_vision_model(CONFIG, quick_gelu=True)
    save_file(
        {f"visual.{key}": value for key, value in model.state_dict().items()},
        str(checkpoint),
    )
    spec = CheckpointSpec(repo_id="pyvisim-tests/tiny-clip", quick_gelu=True)
    monkeypatch.setitem(registry_module.MODEL_CONFIGS, VARIANT, CONFIG)
    monkeypatch.setitem(registry_module.PRETRAINED, VARIANT, {TAG: spec})
    monkeypatch.setattr(
        clip_embedder_module,
        "fetch_checkpoint",
        lambda variant, pretrained, cache_dir=None: checkpoint,
    )
    return VARIANT
