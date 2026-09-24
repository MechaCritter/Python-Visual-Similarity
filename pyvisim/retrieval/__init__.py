"""
Image similarity retrieval.

Storing image embeddings in a search index, retrieving the most similar gallery
images for a query and re-ranking the retrieved candidates.
"""

from . import data, image_store, reranking

__all__ = ["data", "image_store", "reranking"]
