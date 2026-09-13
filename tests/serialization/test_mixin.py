"""Tests for the file contract ``SerializerMixin`` declares and enforces."""

from typing import Any

import pytest

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.serialization import SerializerMixin


def test_a_class_that_writes_files_must_declare_the_contract() -> None:
    """A class implementing both halves of the state contract carries the format."""
    with pytest.raises(TypeError, match="does not declare"):

        class Incomplete(SerializerMixin):
            def to_dict(self) -> dict[str, Any]:
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

    class Parent(SerializerMixin):
        __file_format__ = ".demo"
        __metadata_key__ = "demo"
        __class_key__ = "demo_class"
        __format_version__ = 1
        __state_keys__ = frozenset({"demo_class"})

        def to_dict(self) -> dict[str, Any]:
            return {"demo_class": type(self).__name__}

        @classmethod
        def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "Parent":
            return cls()

    class Child(Parent):
        __format_version__ = 2

    assert Child.__file_format__ == ".demo"
    assert Child.__format_version__ == 2


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


@pytest.mark.parametrize(
    "serialisable",
    [VLADEmbedder, FisherVectorEmbedder, Pipeline, InMemoryImageEmbeddingStore],
)
def test_the_class_key_is_part_of_the_required_state_keys(
    serialisable: type[SerializerMixin],
) -> None:
    """Validation rejects a foreign file, so the key naming its writer is required."""
    assert serialisable.__class_key__ in serialisable.__state_keys__
