import numpy as np

from ..metrics.base_metrics import DescriptorBasedMetrics, SimilarityMetric

class Pipeline(SimilarityMetric):
    """
    A pipeline for computing feature vectors using a set of
    descriptor-based metrics (e.g., VLAD, Fisher, etc.).

    :param metrics: A list of DescriptorBasedMetrics (or anything
                    implementing `compute_vector(image) -> np.ndarray`).
    :param flatten: If True, concatenates all feature vectors
                           into a single 1D array. If False,
                           returns a tuple of them.
    """

    def __init__(
        self,
        metrics: list[DescriptorBasedMetrics],
        flatten: bool = False
    ):
        self.metrics = metrics
        self.flatten = flatten

    def compute_vectors(self, image) -> tuple[np.ndarray, ...] or np.ndarray:
        """
        Compute the feature vectors for the given image using each metric in the pipeline.

        :param image: Input image (NumPy array).
        :return: Tuple of feature vectors, or a single flattened vector if flatten_output=True.
        """
        vectors = []
        for metric in self.metrics:
            vec = metric.compute_vector(image)
            vectors.append(vec)

        if self.flatten:
            return np.concatenate(vectors, axis=0)
        else:
            return tuple(vectors)

    def compare(self, image1: np.ndarray, image2: np.ndarray) -> float:
        pass