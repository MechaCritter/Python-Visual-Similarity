"""Fast unit tests for :class:`pyvisim.encoders.CLIPEncoder`.

These exercise logic that needs no model weights, so they stay in the fast
suite: device resolution and the lazy optional-dependency guard.
"""

from __future__ import annotations

import pytest
import torch

from pyvisim.encoders import CLIPEncoder
from pyvisim.encoders import clip as clip_module
from pyvisim.lazy_import import OptionalImport

# §1 device resolution


def test_resolve_device_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without CUDA, both auto-select and an explicit request yield ``cpu``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert clip_module._resolve_device(None) == "cpu"
    assert clip_module._resolve_device("cuda") == "cpu"
    assert clip_module._resolve_device("cpu") == "cpu"


def test_resolve_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With CUDA available, auto-select and an explicit request yield ``cuda``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert clip_module._resolve_device(None) == "cuda"
    assert clip_module._resolve_device("cuda") == "cuda"


# §2 lazy optional-dependency guard


def test_missing_open_clip_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing a ``CLIPEncoder`` without open_clip raises a helpful error.

    open_clip is installed in the test environment, so a missing dependency is
    simulated by swapping the module-level :class:`OptionalImport` for one whose
    import failed. The constructor must surface the ``pip install`` hint and not
    attempt to build a model.
    """
    broken = OptionalImport(package="open_clip_torch", extra="nn")
    with broken:
        import pyvisim_missing_open_clip_dependency  # type: ignore[import-not-found] # noqa: F401
    assert not broken.is_available
    monkeypatch.setattr(clip_module, "_open_clip_import", broken)
    with pytest.raises(ImportError, match=r"pyvisim\[nn\]"):
        CLIPEncoder()
