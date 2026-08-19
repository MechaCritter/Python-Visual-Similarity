"""
Class-name dispatch for (de)serialising embedders.

This module is used
by :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore` to persist the
embedder alongside the gallery embeddings.
"""

from __future__ import annotations

from typing import Any

from ..typing import Embedder


def embedder_to_dict(embedder: Embedder) -> dict[str, Any]:
    """
    Serialise an embedder into a JSON-safe state dictionary.

    :param embedder: A :class:`~pyvisim.encoders.ImageEmbedderBase` subclass
        instance or a :class:`~pyvisim.encoders.Pipeline`.
    :return: The embedder's ``to_dict`` output.
    :raises TypeError: If ``embedder`` does not expose a ``to_dict`` method.
    """
    to_dict = getattr(embedder, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"Embedder of type {type(embedder).__name__!r} is not serialisable; "
            "it must implement a 'to_dict' method."
        )
    return dict(to_dict())


def embedder_from_dict(state: dict[str, Any]) -> Embedder:
    """
    Rebuild an embedder from a dictionary produced by :func:`embedder_to_dict`.

    :param state: A serialised embedder description with an ``embedder_class`` key.
    :return: The reconstructed embedder instance.
    :raises ValueError: If ``state`` lacks a known ``embedder_class``.
    """
    # Imported lazily so this leaf module never feeds back into the embedder
    # package's import graph at module-load time.
    from .fisher_vector import FisherVectorEmbedder
    from .pipeline import Pipeline
    from .vlad import VLADEmbedder

    # ``Any`` because the registered classes share a ``from_dict`` classmethod
    # that the structural ``Embedder`` protocol does not declare.
    registry: dict[str, Any] = {
        "VLADEmbedder": VLADEmbedder,
        "FisherVectorEmbedder": FisherVectorEmbedder,
        "Pipeline": Pipeline,
    }
    embedder_class = state.get("embedder_class")
    if embedder_class not in registry:
        raise ValueError(
            f"Cannot reconstruct embedder of class {embedder_class!r}. "
            f"Known classes are: {sorted(registry)}."
        )
    embedder: Embedder = registry[embedder_class].from_dict(state)
    return embedder
