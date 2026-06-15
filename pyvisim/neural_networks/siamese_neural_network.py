from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from .._base_classes import SimilarityMetric
from .._utils import cosine_similarity

MatLike = np.ndarray | torch.Tensor | npt.ArrayLike


# TODO: uncomment all "ignore" comments after implementing the class properly.


class SiameseNeuralNetwork(torch.nn.Module, SimilarityMetric):
    def __init__(
        self,
        backbone: torch.nn.Module,
        embedding_dim: int,
        similarity_func: Callable[[MatLike, MatLike], MatLike] = cosine_similarity,  # type: ignore
        device: str | torch.device = "cpu",
    ) -> None:
        self._backbone: torch.nn.Module = backbone
        self._head: torch.nn.Module = torch.nn.Linear(
            int(backbone.output_dim),  # type: ignore[arg-type]
            embedding_dim,
        )
        raise NotImplementedError("SiameseNeuralNetwork is not implemented yet.")

    def encode(
        self,
        image: MatLike,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> np.ndarray:
        raise NotImplementedError("SiameseNeuralNetwork is not implemented yet.")

    def similarity_score(  # type: ignore
        self,
        image_a: MatLike,
        image_b: MatLike,
        dims_a: str = "HWC",
        dims_b: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ) -> float:
        raise NotImplementedError("SiameseNeuralNetwork is not implemented yet.")

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    @head.setter
    def head(self, new_head: torch.nn.Module) -> None:
        if not isinstance(new_head, torch.nn.Module):
            raise ValueError(
                f"Expected new_head to be an instance of torch.nn.Module, got {type(new_head)}"
            )
        if new_head.in_features != self._backbone.output_dim:
            raise ValueError(
                f"Expected new_head to have in_features equal to {self._backbone.output_dim}, got {new_head.in_features}"
            )
        self._head = new_head
