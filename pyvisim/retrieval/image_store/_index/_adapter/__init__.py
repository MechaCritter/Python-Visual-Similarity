"""Adapter letting the image store search through FAISS indexes."""

from .faiss_adapter import FaissIndexAdapter, is_faiss_index

__all__ = ["FaissIndexAdapter", "is_faiss_index"]
