"""
pyvisim: A Python library for image similarity analysis using Image Embedders and Neural Networks.
"""

import importlib
import logging
from types import ModuleType

#: Submodules that import without the optional ``nn`` extra.
__all__ = [
    "base",
    "classic",
    "distance",
    "eval",
    "features",
    "image_store",
    "pixelwise",
    "serialization",
    "structural",
    "typing",
]

#: Submodules that need the optional ``nn`` extra.
_NN_SUBMODULES = ("datasets", "neural_networks")

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name: str) -> ModuleType:
    """
    Imports a submodule the first time it is accessed as an attribute.

    :param name: Name of the accessed attribute.
    :return: The imported submodule.
    :raises AttributeError: If ``name`` is not a submodule of pyvisim.
    """
    if name in __all__ or name in _NN_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
