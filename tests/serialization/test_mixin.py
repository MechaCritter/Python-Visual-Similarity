"""Tests for the file contract ``SerializerMixin`` declares and enforces."""

from pathlib import Path
from typing import Any

import pytest

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.serialization import SerializerMixin, save_state


class _Demo(SerializerMixin):
    """Smallest class carrying the full file contract."""

    __file_format__ = ".demo"
    __metadata_key__ = "demo"
    __class_key__ = "demo_class"
    __format_version__ = 1
    __state_keys__ = frozenset({"value"})

    def _state(self) -> dict[str, Any]:
        return {"value": 1}

    @classmethod
    def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "_Demo":
        return cls()


def test_a_class_that_writes_files_must_declare_the_contract() -> None:
    """A class implementing both halves of the state contract carries the format."""
    with pytest.raises(TypeError, match="does not declare"):

        class Incomplete(SerializerMixin):
            def _state(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "Incomplete":
                return cls()


def test_a_class_that_writes_no_file_yet_declares_nothing() -> None:
    """An intermediate base leaves the format to the class completing it."""

    class Intermediate(SerializerMixin):
        pass

    assert not hasattr(Intermediate, "__file_format__")


def test_a_declared_contract_reaches_the_subclasses() -> None:
    """A subclass overrides what differs and inherits the rest."""

    class Child(_Demo):
        __format_version__ = 2

    assert Child.__file_format__ == ".demo"
    assert Child.__format_version__ == 2


@pytest.mark.parametrize("stamped_key", ["format_version", "demo_class"])
def test_a_file_without_a_stamped_key_is_not_valid(
    stamped_key: str, tmp_path: Path
) -> None:
    """The keys the mixin writes are required on load, without being listed."""
    state = _Demo().to_dict()
    del state[stamped_key]
    save_state(state, path := tmp_path / "file.demo", _Demo.__metadata_key__)
    with pytest.raises(ValueError, match="not a valid .demo file"):
        _Demo.load_from_disk(path)


@pytest.mark.parametrize(
    "serialisable",
    [VLADEmbedder, FisherVectorEmbedder, Pipeline, InMemoryImageEmbeddingStore],
)
def test_every_shipped_class_declares_its_own_file_format(
    serialisable: type[SerializerMixin],
) -> None:
    """The classes users save and load describe their files themselves."""
    contract = (
        "__file_format__",
        "__metadata_key__",
        "__class_key__",
        "__format_version__",
        "__state_keys__",
    )
    assert all(getattr(serialisable, name, None) is not None for name in contract)
