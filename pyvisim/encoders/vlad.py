from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..features._features import FeatureExtractorBase, RootSIFT
from ..encoders._base_encoder import ImageEncoderBase
from .._utils import cosine_similarity

# TODO: Add an enum that initialized the clustering models with weights trained on the Oxford Flowers dataset or similar 8specify also whether PCA was used or not)
class VLADEncoder(ImageEncoderBase):
    """
    This class encodes images into VLAD descriptor vectors
    using a chosen feature extractor and a pretrained K-Means model,
    then compares two VLAD descriptor vectors with a user-specified
    or default (cosine) similarity function.

    The output when calling `compute_vector` has shape (num_clusters * feature_dim,).

    You can use euclidean distance, manhattan distance, etc. as the similarity function.

    The encoding can be used for indexing, retrieval, clustering or classification tasks.
    :param feature_extractor: Feature extractor instance (should implement __call__).
    :param kmeans_model: KMeans model instance from scikit-learn.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
    :param pca: PCA model for dimensionality reduction (optional).
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.

    References:
    ==========
    [1] Relja Arandjelović and Andrew Zisserman, 'All About VLAD', Department of Engineering Science, University of Oxford.
    [2] Relja Arandjelović and Andrew Zisserman, "Three things everyone should know to improve object retrieval," Department of Engineering Science, University of Oxford.
    [3] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """
    def __init__(
            self,
            feature_extractor: FeatureExtractorBase=RootSIFT(),
            weights=None,
            kmeans_model: KMeans=None,
            power_norm_weight: float = 1, # no paper found where power norm weight is used for VLAD
            norm_order: int = 2,
            epsilon: float = 1e-9,
            flatten: bool = True,
            similarity_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
            pca: PCA = None,
            raise_error_when_pca_incompatible: bool = False) -> None:
        if kmeans_model is not None:
            if not isinstance(kmeans_model, KMeans):
                raise ValueError(f"The clustering model must be an instance of KMeans, not {type(kmeans_model)}")
        if weights is not None:
            if (weights_class:=weights.__class__.__name__) != 'KMeansWeights':
                raise ValueError(f"You can only pass an instance of KMeansWeights, not {weights_class}")
        super().__init__(feature_extractor,
                         weights,
                         kmeans_model,
                         similarity_func,
                         power_norm_weight,
                         norm_order,
                         epsilon,
                         flatten,
                         pca,
                         raise_error_when_pca_incompatible)

    @property
    def clustering_model(self):
        return ImageEncoderBase.clustering_model.fget(self)

    @clustering_model.setter
    def clustering_model(self, model):
        if not isinstance(model, KMeans):
            raise ValueError(f"The clustering model must be an instance of KMeans, not {type(model)}")
        ImageEncoderBase.clustering_model.fset(self, model)

    def encode(self, images: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        all_encodings = []
        if isinstance(images, torch.Tensor):
            raise RuntimeError("Torch images are not supported yet.")
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
