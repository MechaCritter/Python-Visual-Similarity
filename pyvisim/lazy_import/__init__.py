"""Lazy-import helpers for pyvisim's optional dependencies."""

from .lazy_import import OptionalImport
from .torch_backend import is_tensor, torch_import

__all__ = [
    "OptionalImport",
    "is_tensor",
    "torch_import",
]
