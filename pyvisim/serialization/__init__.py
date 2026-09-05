"""safetensors-backed serialization for image embedders."""

from .mixin import SerializerMixin
from .serialization import (
    EMBEDDER_METADATA_KEY,
    decode_array_node,
    embedder_from_dict,
    embedder_to_dict,
    load_state,
    save_state,
)

__all__ = [
    "EMBEDDER_METADATA_KEY",
    "SerializerMixin",
    "decode_array_node",
    "embedder_from_dict",
    "embedder_to_dict",
    "load_state",
    "save_state",
]
