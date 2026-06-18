"""Accelerated image indexes and retrieval over an image gallery."""

from ._base_index import ImageIndex
from .image_index import ImageIndexIVFFlat

__all__ = ["ImageIndex", "ImageIndexIVFFlat"]
