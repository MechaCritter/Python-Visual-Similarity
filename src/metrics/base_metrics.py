"""
Defines the abstract base class for similarity metrics.
"""

import abc
import logging
from typing import Callable, Optional, Iterable

import numpy as np
from sklearn.decomposition import PCA

from src.utils import cosine_similarity
from src.config import setup_logging

setup_logging()


class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity metrics.

    All concrete similarity metric classes must inherit from this class.
    """
    _logger = logging.getLogger('Similarity_Metrics')
    @abc.abstractmethod
    def compare(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute a similarity score between two images.

        :param image1: First image
        :param image2: Second image
        :return: A similarity score
        """
        pass

class DescriptorBasedMetrics(SimilarityMetric):
    """
    Base class for descriptor-based similarity metrics (e.g., VLAD, Fisher Vectors). These metrics
    compute a single vector representation for an image using a feature extractor and a clustering model.

    This class provides core functionality to extract features, normalize,
    reduce dimensions, and compute similarity metrics between descriptor vectors.

    Attributes:
        feature_extractor: A feature extractor instance (should implement __call__).
        clustering_model: A clustering model used for computing the descriptors.
        power_norm_weight: Exponent for power normalization (default: 0.5).
        norm_order: Norm order for vector normalization (default: 2 -> L2).
        epsilon: Small value to prevent division by zero in normalization.
        flatten: Whether to flatten the computed descriptor vector.
        similarity_func: A callable for computing similarity between two vectors (default: None).
        pca: PCA model for dimensionality reduction (optional).
    """
    def __init__(
            self,
            feature_extractor,
            clustering_model,
            power_norm_weight: float = 0.5,
            norm_order: int = 2,
            epsilon: float = 1e-9,
            flatten: bool = True,
            similarity_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
            pca: Optional[PCA] = None
    ):
        """
        Initializes the DescriptorBasedMetrics instance.

        :param feature_extractor: Feature extractor instance (should implement __call__).
        :param clustering_model: Clustering model used for generating descriptors.
        :param power_norm_weight: Exponent for power normalization (default: 0.5).
        :param norm_order: Norm order for normalization (default: 2).
        :param epsilon: Small constant to avoid division by zero.
        :param flatten: Whether to flatten the computed descriptor vector (default: True).
        :param similarity_func: Function for computing similarity (default: None).
        :param pca: PCA model for dimensionality reduction (optional).
        """
        self.feature_extractor = feature_extractor
        self.clustering_model = clustering_model
        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.similarity_func = similarity_func if similarity_func else cosine_similarity
        self.pca = pca

    @abc.abstractmethod
    def compute_vector(self, image: np.ndarray) -> np.ndarray:
        """Computes feature vector for an image."""
        pass

    def fit(self, images: Iterable[np.ndarray], reduce_dimension: bool = False) -> None:
        """
        Fits the clustering model using features extracted from a list of images. The first element
        of the iterable has to be the image.

        :param images: An iterable of images
        :param reduce_dimension: Whether to apply PCA for dimensionality reduction
        :param reduction_factor: Factor for dimensionality reduction
        """
        features = np.vstack([self.feature_extractor(image) for image, *_ in images])
        if reduce_dimension:
            if self.pca is None:
                raise ValueError("PCA is not initialized for dimensionality reduction. Please train your PCA model first.")
            features = self.pca.transform(features)
        self.clustering_model.fit(features)

    def fit_pca(self, images: Iterable[np.ndarray], n_components: int) -> None:
        """
        Fits the PCA model using features extracted from a list of images. The first element
        of the iterable has to be the image.

        :param images: An iterable of images
        :param n_components: Number of components for PCA
        """
        features = np.vstack([self.feature_extractor(image) for image in images])
        self.pca = PCA(n_components=n_components).fit(features)

    def compare(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes descriptor vectors for two images and compares them.

        :param image1: First image
        :param image2: Second image
        :return: Similarity score
        """
        vector1 = self.compute_vector(image1)
        vector2 = self.compute_vector(image2)
        result = self.similarity_func(vector1, vector2) if self.similarity_func else cosine_similarity(vector1, vector2)[0][0]
        return float(result)

    def __repr__(self) -> str:
        n_clusters = None
        if self.clustering_model:
            if hasattr(self.clustering_model, 'n_clusters'):
                n_clusters = self.clustering_model.n_clusters
            elif hasattr(self.clustering_model, 'n_components'):
                n_clusters = self.clustering_model.n_components
        return self.__class__.__name__ + f"(feature_extractor={self.feature_extractor.__class__.__name__}, " \
               f"similarity_func={self.similarity_func.__name__}, " \
               f"Number of clusters/components={n_clusters}, " \
                f"power_norm_weight={self.power_norm_weight}, " \
                f"norm_order={self.norm_order}"




