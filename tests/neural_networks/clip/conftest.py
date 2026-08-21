"""Fixtures shared by the :mod:`pyvisim.neural_networks.clip` tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyvisim.neural_networks import ClipEmbedder

from ._tiny_clip import TAG, register_tiny_variant


@pytest.fixture()
def tiny_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Register a tiny CLIP variant whose checkpoint lives in ``tmp_path``."""
    return register_tiny_variant(monkeypatch, tmp_path)


@pytest.fixture()
def embedder(tiny_variant: str) -> ClipEmbedder:
    """A ``ClipEmbedder`` built on the tiny variant."""
    return ClipEmbedder(tiny_variant, TAG, device="cpu")
