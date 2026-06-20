"""
Class-name dispatch for (de)serialising encoders.

This module is used
by :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore` to persist the
encoder alongside the gallery embeddings.
"""

from __future__ import annotations

from typing import Any

from ..typing import Encoder


def encoder_to_dict(encoder: Encoder) -> dict[str, Any]:
    """
    Serialise an encoder into a JSON-safe state dictionary.

    :param encoder: A :class:`~pyvisim.encoders.ImageEncoderBase` subclass
        instance or a :class:`~pyvisim.encoders.Pipeline`.
    :return: The encoder's ``to_dict`` output.
    :raises TypeError: If ``encoder`` does not expose a ``to_dict`` method.
    """
    to_dict = getattr(encoder, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"Encoder of type {type(encoder).__name__!r} is not serialisable; "
            "it must implement a 'to_dict' method."
        )
    return dict(to_dict())


def encoder_from_dict(state: dict[str, Any]) -> Encoder:
    """
    Rebuild an encoder from a dictionary produced by :func:`encoder_to_dict`.

    :param state: A serialised encoder description with an ``encoder_class`` key.
    :return: The reconstructed encoder instance.
    :raises ValueError: If ``state`` lacks a known ``encoder_class``.
    """
    # Imported lazily so this leaf module never feeds back into the encoder
    # package's import graph at module-load time.
    from .fisher_vector import FisherVectorEncoder
    from .pipeline import Pipeline
    from .vlad import VLADEncoder

    # ``Any`` because the registered classes share a ``from_dict`` classmethod
    # that the structural ``Encoder`` protocol does not declare.
    registry: dict[str, Any] = {
        "VLADEncoder": VLADEncoder,
        "FisherVectorEncoder": FisherVectorEncoder,
        "Pipeline": Pipeline,
    }
    encoder_class = state.get("encoder_class")
    if encoder_class not in registry:
        raise ValueError(
            f"Cannot reconstruct encoder of class {encoder_class!r}. "
            f"Known classes are: {sorted(registry)}."
        )
    encoder: Encoder = registry[encoder_class].from_dict(state)
    return encoder
