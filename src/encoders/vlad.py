"""
Implementation of the VLAD similarity metric using a user-supplied feature extractor
and a pretrained K-Means model.
"""

from typing import Callable, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.features._features import FeatureExtractorBase
from src.encoders._base_encoder import ImageEncoderBase
from src.utils import cosine_similarity


class VLADEncoder(ImageEncoderBase):
    """
    # TODO: cite paper from Jégou et al. (2010)
    This class encodes images into VLAD descriptor vectors
    using a chosen feature extractor and a pretrained K-Means model,
    then compares two VLAD descriptor vectors with a user-specified
    or default (cosine) similarity function.

    The output when calling `compute_vector` has shape (num_clusters * feature_dim,).

    You can use euclidean distance, manhattan distance, etc. as the similarity function.

    :param feature_extractor: An instance of FeatureExtractorBase
    :param kmeans_model: Pretrained K-Means model (with `.predict` and `.cluster_centers_`)
    :param power_norm_weight: Exponent for power normalization (default 0.5)
    :param norm_order: Norm order for final normalization (default 2 -> L2)
    :param epsilon: Small constant to avoid division by zero
    :param flatten: Whether to flatten the final VLAD vector
    :param similarity_func: A function(vec1, vec2) -> float for computing similarity
    :param pca: PCA transformer for optional dimensionality reduction of descriptors (highly recommended, since
    feature vectors are usually high-dimensional but sparse)

    References:
    ==========

    """
    def __init__(
            self,
            feature_extractor: FeatureExtractorBase,
            kmeans_model: KMeans,
            power_norm_weight: float = 1, # no paper found where power norm weight is used for VLAD
            norm_order: int = 2,
            epsilon: float = 1e-9,
            flatten: bool = True,
            similarity_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
            pca: PCA = None
    ):
        if not isinstance(kmeans_model, KMeans):
            raise ValueError(f"The clustering model must be an instance of KMeans, not {type(kmeans_model)}")
        super().__init__(feature_extractor,
                         kmeans_model,
                         similarity_func,
                         power_norm_weight,
                         norm_order,
                         epsilon,
                         flatten,
                         pca)

    def encode(self, images: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        all_encodings = []
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images] # Handle single image case
        for image in images:
            descriptors = self.feature_extractor(image)
            if self.pca:
                descriptors = self.pca.transform(descriptors.astype(np.float32))

            if descriptors is None or descriptors.shape[0] == 0:
                return np.zeros(len(self.clustering_model.cluster_centers_) * descriptors.shape[1], dtype=np.float32)

            labels = self.clustering_model.predict(descriptors.astype(np.float32))
            centroids = self.clustering_model.cluster_centers_

            k = len(centroids)
            dim = descriptors.shape[1]
            descriptor_vector = np.zeros((k, dim), dtype=np.float32)

            for i, desc in enumerate(descriptors):
                cluster_id = labels[i]
                descriptor_vector[cluster_id] += (desc - centroids[cluster_id])

            descriptor_vector = np.sign(descriptor_vector) * np.abs(descriptor_vector) ** self.power_norm_weight
            norms = np.linalg.norm(descriptor_vector, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
            descriptor_vector = descriptor_vector / norms

            if self.flatten:
                descriptor_vector = descriptor_vector.flatten()

            all_encodings.append(descriptor_vector)

        return np.vstack(all_encodings)
