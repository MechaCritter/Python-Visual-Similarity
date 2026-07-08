"""Fast unit tests for the CLIP checkpoint registry and download cache.

These tests never touch the network: downloads are exercised against stubbed
HTTP responses and tiny payloads, so they stay in the fast suite.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyvisim._errors import ChecksumMismatchError
from pyvisim.neural_networks.clip import (
    CHECKPOINTS,
    CheckpointSpec,
    available_variants,
    fetch_checkpoint,
    get_checkpoint_spec,
)
from pyvisim.neural_networks.clip import _checkpoints as checkpoints_module

#: Tiny stand-in for a checkpoint archive.
_PAYLOAD = b"pyvisim clip checkpoint payload"
_PAYLOAD_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()

#: Variant name registered on top of the real registry for cache tests.
_TEST_VARIANT = "test-variant"


class _FakeResponse:
    """Minimal stand-in for a streaming :class:`requests.Response`."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        yield self._payload


@pytest.fixture()
def payload_variant(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a tiny test checkpoint spec and hand back its variant name."""
    spec = CheckpointSpec(
        filename="tiny.pt",
        sha256=_PAYLOAD_SHA256,
        image_size=224,
        embedding_dim=512,
    )
    monkeypatch.setitem(checkpoints_module.CHECKPOINTS, _TEST_VARIANT, spec)
    return _TEST_VARIANT


# §1 registry contents


def test_registry_lists_exactly_the_supported_variants() -> None:
    """The registry covers ViT-B/32, ViT-B/16 and ViT-L/14 and nothing else."""
    assert set(available_variants()) == {"ViT-B/32", "ViT-B/16", "ViT-L/14"}


@pytest.mark.parametrize("variant", ["ViT-B/32", "ViT-B/16", "ViT-L/14"])
def test_specs_are_self_consistent(variant: str) -> None:
    """Every spec carries a full SHA-256 digest that its URL embeds.

    OpenAI serves each checkpoint under a path segment equal to the SHA-256
    digest of the file, so the digest in the spec and the digest in the URL
    must be one and the same value.
    """
    spec = get_checkpoint_spec(variant)
    assert len(spec.sha256) == 64
    assert set(spec.sha256) <= set("0123456789abcdef")
    assert spec.url == (
        f"https://openaipublic.azureedge.net/clip/models/{spec.sha256}/{spec.filename}"
    )


def test_specs_match_the_published_model_configuration() -> None:
    """Input sizes and embedding widths match the official model configs."""
    assert all(spec.image_size == 224 for spec in CHECKPOINTS.values())
    assert CHECKPOINTS["ViT-B/32"].embedding_dim == 512
    assert CHECKPOINTS["ViT-B/16"].embedding_dim == 512
    assert CHECKPOINTS["ViT-L/14"].embedding_dim == 768


def test_get_checkpoint_spec_unknown_variant_raises() -> None:
    """An unknown variant name raises a ``ValueError`` naming the variant."""
    with pytest.raises(ValueError, match="RN50"):
        get_checkpoint_spec("RN50")


# §2 cache behaviour


def test_fetch_returns_verified_cached_file_without_downloading(
    payload_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file with a matching digest is returned without any download."""
    cached = tmp_path / "tiny.pt"
    cached.write_bytes(_PAYLOAD)
    monkeypatch.setattr(
        checkpoints_module,
        "_download_checkpoint",
        lambda spec, target: pytest.fail("download must not run on a cache hit"),
    )
    assert fetch_checkpoint(payload_variant, cache_dir=tmp_path) == cached


def test_fetch_downloads_when_cache_is_empty(
    payload_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing cache entry triggers a download into the cache directory."""
    monkeypatch.setattr(
        checkpoints_module,
        "_download_checkpoint",
        lambda spec, target: target.write_bytes(_PAYLOAD),
    )
    path = fetch_checkpoint(payload_variant, cache_dir=tmp_path)
    assert path == tmp_path / "tiny.pt"
    assert path.read_bytes() == _PAYLOAD


def test_fetch_corrupted_cache_warns_and_redownloads(
    payload_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file failing its digest check is warned about and replaced."""
    cached = tmp_path / "tiny.pt"
    cached.write_bytes(b"corrupted bytes")
    monkeypatch.setattr(
        checkpoints_module,
        "_download_checkpoint",
        lambda spec, target: target.write_bytes(_PAYLOAD),
    )
    with pytest.warns(FutureWarning, match="SHA-256"):
        path = fetch_checkpoint(payload_variant, cache_dir=tmp_path)
    assert path.read_bytes() == _PAYLOAD


def test_fetch_rejects_non_regular_file_target(
    payload_variant: str, tmp_path: Path
) -> None:
    """A directory obstructing the cache path raises an ``OSError``."""
    (tmp_path / "tiny.pt").mkdir()
    with pytest.raises(OSError, match="not a regular file"):
        fetch_checkpoint(payload_variant, cache_dir=tmp_path)


# §3 download verification


def test_download_verifies_sha256_and_installs_file(
    payload_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download whose digest matches is moved into the cache atomically."""
    monkeypatch.setattr(
        checkpoints_module.requests,
        "get",
        lambda url, stream, timeout: _FakeResponse(_PAYLOAD),
    )
    path = fetch_checkpoint(payload_variant, cache_dir=tmp_path)
    assert path.read_bytes() == _PAYLOAD
    assert [entry.name for entry in tmp_path.iterdir()] == ["tiny.pt"]


def test_download_checksum_mismatch_raises_and_leaves_no_file(
    payload_variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tampered download bytes raise and leave the cache directory clean.

    Neither the final file nor a partial temporary file may survive, so a
    later call cannot pick up unverified bytes.
    """
    monkeypatch.setattr(
        checkpoints_module.requests,
        "get",
        lambda url, stream, timeout: _FakeResponse(b"tampered payload"),
    )
    with pytest.raises(ChecksumMismatchError, match=_PAYLOAD_SHA256):
        fetch_checkpoint(payload_variant, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
