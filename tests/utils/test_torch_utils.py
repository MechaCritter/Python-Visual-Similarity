"""Unit tests for :mod:`pyvisim.utils.torch_utils`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyvisim.utils.torch_utils import (
    decode_state_dict,
    encode_state_dict,
    resolve_device,
)

# §1 device resolution


def test_resolve_device_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without CUDA, both auto-select and an explicit request yield ``cpu``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == "cpu"
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cuda:1") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With CUDA available, auto-select and an explicit request yield ``cuda``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_accepts_torch_device() -> None:
    """A :class:`torch.device` is accepted and rendered as its name."""
    assert resolve_device(torch.device("cpu")) == "cpu"


# §2 state-dict encoding


def test_encoded_state_dict_holds_one_array_node_per_entry() -> None:
    """Every ``state_dict`` entry becomes an embedded array node."""
    module = torch.nn.Linear(3, 2)
    encoded = encode_state_dict(module)
    assert set(encoded) == set(module.state_dict())
    assert encoded["weight"]["__ndarray__"] is True
    assert encoded["weight"]["shape"] == [2, 3]
    assert encoded["weight"]["dtype"] == "float32"


def test_decode_restores_the_exact_tensors() -> None:
    """Decoding the restored arrays reproduces the original tensors exactly."""
    module = torch.nn.Linear(3, 2)
    encoded = encode_state_dict(module)
    restored = {name: node["data"] for name, node in encoded.items()}
    decoded = decode_state_dict(restored)
    for name, tensor in module.state_dict().items():
        assert torch.equal(decoded[name], tensor)


def test_decode_copies_read_only_arrays() -> None:
    """Read-only arrays (as memory-mapped by safetensors) are copied first."""
    array = np.zeros((2, 3), dtype=np.float32)
    array.flags.writeable = False
    decoded = decode_state_dict({"weight": array})
    assert decoded["weight"].shape == (2, 3)

    decoded["weight"][0, 0] = 1.0
    assert array[0, 0] == 0.0
