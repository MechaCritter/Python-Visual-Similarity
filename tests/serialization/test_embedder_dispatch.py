"""Tests for rebuilding an embedder of any class through ``SerializableImageEmbedder.from_dict``."""

from typing import Any, ClassVar

import numpy as np
import pytest

from pyvisim.base import SerializableImageEmbedder
from pyvisim.classic import VLADEmbedder
from pyvisim.typing import FloatNumpyArray, UInt8NumpyArray


class _Registered(SerializableImageEmbedder):
    """Smallest embedder defined outside the library."""

    __format_version__: ClassVar[int] = 1

    def _embed(self, images: list[UInt8NumpyArray]) -> FloatNumpyArray:
        return np.ones((len(images), 2))

    def _state(self) -> dict[str, Any]:
        return {
            "similarity_func": self.similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "_Registered":
        return cls(batch_size=state["batch_size"])


class _WithoutFromDict(SerializableImageEmbedder):
    """An embedder that describes itself but cannot be rebuilt."""

    __format_version__: ClassVar[int] = 1

    def _embed(self, images: list[UInt8NumpyArray]) -> FloatNumpyArray:
        return np.ones((len(images), 2))

    def _state(self) -> dict[str, Any]:
        return {}


def test_dispatch_rebuilds_the_class_that_wrote_the_state(
    learned_vlad_embedder: VLADEmbedder,
) -> None:
    """The class named in the state rebuilds it."""
    rebuilt = SerializableImageEmbedder.from_dict(learned_vlad_embedder.to_dict())
    assert isinstance(rebuilt, VLADEmbedder)


def test_a_subclass_registers_itself_on_definition() -> None:
    """An embedder class needs no registry entry to be found by name."""
    rebuilt = SerializableImageEmbedder.from_dict(_Registered(batch_size=3).to_dict())
    assert isinstance(rebuilt, _Registered)
    assert rebuilt.batch_size == 3


@pytest.mark.parametrize("name", ["NoSuchEmbedder", "ClusteringBasedEmbedder", None])
def test_a_state_naming_no_concrete_class_is_rejected(name: str | None) -> None:
    """Unknown and abstract class names cannot be rebuilt."""
    with pytest.raises(ValueError, match="Cannot reconstruct embedder"):
        SerializableImageEmbedder.from_dict({"embedder_class": name})


def test_a_subclass_without_from_dict_raises() -> None:
    """The dispatch never calls back into itself for a subclass."""
    with pytest.raises(NotImplementedError, match="_WithoutFromDict"):
        SerializableImageEmbedder.from_dict(_WithoutFromDict().to_dict())
