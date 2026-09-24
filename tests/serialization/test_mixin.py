"""Tests for the file contract ``SerializerMixin`` declares and enforces."""

from pathlib import Path
from typing import Any

import pytest

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder
from pyvisim.features import SIFT, DeepConvFeature, RootSIFT
from pyvisim.image_store import InMemoryImageEmbeddingStore
from pyvisim.serialization import SerializerMixin, save_state


class _Demo(SerializerMixin):
    """Smallest class carrying the full file contract."""

    __file_format__ = ".demo"
    __metadata_key__ = "demo"
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


@pytest.mark.parametrize("stamped_key", ["format_version", "__class__"])
def test_a_file_without_a_stamped_key_is_not_valid(
    stamped_key: str, tmp_path: Path
) -> None:
    """The keys the mixin writes are required on load, without being listed."""
    state = _Demo().to_dict()
    del state[stamped_key]
    save_state(state, path := tmp_path / "file.demo", _Demo.__metadata_key__)
    with pytest.raises(ValueError, match="not a valid .demo file"):
        _Demo.load_from_disk(path)


def test_an_unreadable_file_names_its_kind_and_the_reason(tmp_path: Path) -> None:
    """A file stored under another metadata key is rejected with the cause kept."""
    save_state(_Demo().to_dict(), path := tmp_path / "file.demo", "other")
    with pytest.raises(ValueError, match=r"not a valid \.demo file: .*'demo'"):
        _Demo.load_from_disk(path)


@pytest.mark.parametrize(
    "serializable",
    [
        VLADEmbedder,
        FisherVectorEmbedder,
        Pipeline,
        InMemoryImageEmbeddingStore,
        SIFT,
        RootSIFT,
        DeepConvFeature,
    ],
)
def test_every_shipped_class_declares_its_own_file_format(
    serializable: type[SerializerMixin],
) -> None:
    """The classes users save and load describe their files themselves."""
    contract = (
        "__file_format__",
        "__metadata_key__",
        "__format_version__",
        "__state_keys__",
    )
    assert all(getattr(serializable, name, None) is not None for name in contract)


def test_every_class_in_a_state_is_named_under_the_same_key(
    learned_vlad_embedder: VLADEmbedder,
) -> None:
    """The objects nested in a state name their class the way the outer one does."""
    state = Pipeline([learned_vlad_embedder]).to_dict()
    embedder_state = state["classic"][0]
    assert state["__class__"] == "Pipeline"
    assert embedder_state["__class__"] == "VLADEmbedder"
    assert embedder_state["clustering_model"]["__class__"] == "KMeans"
    assert (
        embedder_state["feature_extractor"]["__class__"]
        == type(learned_vlad_embedder.feature_extractor).__name__
    )
