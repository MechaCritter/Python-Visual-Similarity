import abc
import logging

import numpy as np

from ._utils import is_numpy_image

class SimilarityMetric(abc.ABC):
    """
    Abstract base for all similarity encoders.

    All concrete similarity metric classes must inherit from this class.
    """
    _logger = logging.getLogger('Similarity_Metrics')
    @abc.abstractmethod
    def similarity_score(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute a similarity score between two images.

        :param image1: First image
        :param image2: Second image
        :return: A similarity score
        """
        is_numpy_image(image1, 0)
        is_numpy_image(image2, 1)
        pass

class FeatureExtractorBase(abc.ABC):
    """
    Abstract interface for extracting features from images.

    A feature extractor transforms an image (NumPy array) into a
    set of feature vectors (NumPy array).
    """
    _logger = logging.getLogger("Feature_Extractor")
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts features from an image.

        :param image: Input image (NumPy array).
        :return: Feature descriptors (NumPy array).
        """
        is_numpy_image(image, 0)
        pass

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """
        The dimensionality (D) of each feature vector, i.e., shape[1] of the output.
        """
        pass