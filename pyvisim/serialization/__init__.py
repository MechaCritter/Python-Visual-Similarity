"""safetensors-backed serialization for image embedders."""

from .mixin import SerializerMixin
from .serialization import decode_array_node, load_state, save_state

__all__ = [
    "SerializerMixin",
    "decode_array_node",
    "load_state",
    "save_state",
]
