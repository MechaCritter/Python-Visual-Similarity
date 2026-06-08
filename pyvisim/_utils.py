import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as cs


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between two matrices.

    :param x: First matrix
    :param y: Second matrix

    :return: Cosine similarity matrix
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    x = x.reshape(1, -1) if len(x.shape) == 1 else x
    y = y.reshape(1, -1) if len(y.shape) == 1 else y
    if x.shape[-1] <= 1 or y.shape[-1] <= 1:
        raise ValueError(f"Cosine similarity requires at least 2 features. Got {x.shape[-1]} features for x and {y.shape[-1]} features for y.")

    return cs(x, y)
