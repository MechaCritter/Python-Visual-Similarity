"""
File: similarity_pipeline.py
============================

Defines a pipeline that composes multiple similarity metrics
and compares two images, returning a vector of results.
"""

from typing import List, Optional
import numpy as np

from .base_metrics import SimilarityMetric


class SimilarityPipeline:
    """
    A pipeline that can hold multiple similarity metrics and
    compare two images, returning a list of similarity scores.

    :param metrics: List of similarity metric instances.
    :param reduce: if True, average out all similarity scores and return a single float
    """
    def __init__(
            self,
            metrics: List[SimilarityMetric],
            reduce: Optional[bool] = False
    ):
        self.metrics = metrics
        self.reduce = reduce

    def compare(self, image1, image2) -> float | list[float]:
        scores = [metric.compare(image1, image2) for metric in self.metrics]
        result = float(np.mean(scores)) if self.reduce else scores
        return result

    def __repr__(self) -> str:
        format_str =  self.__class__.__name__ + "("
        for metric in self.metrics:
            format_str += f"\n{metric}"
        format_str += "\n)"
        return format_str


