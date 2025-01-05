"""
Defines the abstract base class for similarity metrics.
"""

import abc
import logging
import warnings
from typing import Callable, Optional, Iterable

import numpy as np
from sklearn.decomposition import PCA

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
        power_norm_weight: Exponent for power normalization (default: 0.5).
        norm_order: Norm order for vector normalization (default: 2 -> L2).
        epsilon: Small value to prevent division by zero in normalization.
        flatten: Whether to flatten the computed descriptor vector.
        similarity_func: A callable for computing similarity between two vectors (default: None).
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
        self.pca = pca

        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.similarity_func = similarity_func

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, value):
        self._feature_extractor = value

    @property
    def clustering_model(self):
        return self._clustering_model

    @clustering_model.setter
    def clustering_model(self, value):
        self._clustering_model = value

        if self._pca:
            if self._pca.n_components != self._clustering_model.n_features_in_:
                warnings.warn(f"PCA is incompatible with the new clustering model. "
                                f"PCA input size: {self._pca.n_components}, "
                                f"New clustering model input size: {self._clustering_model.n_features_in_}. "
                                "PCA will be reset to None to avoid errors.")
                self._pca = None

    @property
    def pca(self):
        return self._pca

    @pca.setter
    def pca(self, value: PCA):
        if not self._clustering_model:
            raise ValueError("PCA cannot be set without an existing clustering model.")

        if value.n_components != self._clustering_model.n_features_in_:
            raise ValueError("PCA input size has to match the clustering model input size."
                             f"PCA model has input size {value.n_components}, "
                             f"while clustering model has input size {self._clustering_model.n_features_in_}")

        self._pca = value

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
        """
        features = np.vstack([self.feature_extractor(image) for image, *_ in images])
        if reduce_dimension:
            if not self._pca:
                raise ValueError("PCA is not initialized for dimensionality reduction. Please train your PCA model first.")
            features = self._pca.transform(features)
        self._clustering_model.fit(features)

    def fit_pca(self, images: Iterable[np.ndarray], n_components: int) -> None:
        """
        Fits the PCA model using features extracted from a list of images. The first element
        of the iterable has to be the image.

        :param images: An iterable of images
        :param n_components: Number of components for PCA
        """
        features = np.vstack([self.feature_extractor(image) for image in images])
        self._pca = PCA(n_components=n_components).fit(features)

    def transform(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """
        Transforms a list of images into descriptor vectors. Prefer using a generator for the images
        to save memory.

        :param images: An iterable of images
        :return: array of shape (num_imgs, vector_size)
        """
        return np.vstack([self.compute_vector(image) for image in images])

    def compare(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes descriptor vectors for two images and compares them.

        :param image1: First image
        :param image2: Second image
        :return: Similarity score
        """
        if image1.shape != image2.shape:
            raise ValueError(f"Images must have the same shape. Got {image1.shape} and {image2.shape} instead.")

        vector1 = self.compute_vector(image1)
        vector2 = self.compute_vector(image2)
        result = self.similarity_func(vector1, vector2)
        return float(result)

    def __repr__(self) -> str:
        n_clusters = None
        if self._clustering_model:
            if hasattr(self._clustering_model, 'n_clusters'):
                n_clusters = self._clustering_model.n_clusters
            elif hasattr(self._clustering_model, 'n_components'):
                n_clusters = self._clustering_model.n_components
        return self.__class__.__name__ + f"(feature_extractor={self.feature_extractor.__class__.__name__}, " \
               f"similarity_func={self.similarity_func.__name__}, " \
               f"Number of clusters/components={n_clusters}, " \
               f"power_norm_weight={self.power_norm_weight}, " \
               f"norm_order={self.norm_order}"





