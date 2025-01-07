import abc
import logging

import numpy as np

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
        pass
