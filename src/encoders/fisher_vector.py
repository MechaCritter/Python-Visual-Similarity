"""
Implementation of the Fisher Vector similarity metric using a user-supplied feature extractor
and a pretrained Gaussian Mixture Model (GMM).
"""

from typing import Callable, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.features._features import FeatureExtractorBase
from src.encoders._base_encoder import ImageEncoderBase
from src.utils import cosine_similarity


class FisherVectorEncoder(ImageEncoderBase):
    """
    This class serves as an encoder that transforms input images into Fisher Vector descriptors.

    The Fisher Vector representation is based on the gradients of the GMM parameters
    (weights, means, and covariances) with respect to the feature descriptors extracted
    from the images. The representation is optionally power-normalized and L2-normalized.

    The output when calling `compute_vector` has shape (2 * num_clusters * feature_dim + num_clusters,).

    :param feature_extractor: An instance of FeatureExtractorBase to extract features from images.
    :param gmm_model: Pretrained Gaussian Mixture Model (GMM) for Fisher Vector computation.
                      Must be an instance of sklearn.mixture.GaussianMixture.
    :param power_norm_weight: Exponent for power normalization (default 0.5).
    :param norm_order: Norm order for final normalization (default 2 -> L2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the final Fisher Vector.
    :param similarity_func: A callable function(vec1, vec2) -> float to compute similarity.
    :param pca: PCA transformer for optional dimensionality reduction of descriptors.
    """
    def __init__(self,
                 feature_extractor: FeatureExtractorBase,
                 gmm_model: GaussianMixture,
                 power_norm_weight: float = 0.5,
                 norm_order: int = 2,
                 epsilon: float = 1e-9,
                 flatten: bool = True,
                 similarity_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
                 pca: PCA = None):
        if not isinstance(gmm_model, GaussianMixture):
            raise ValueError(f"The clustering model must be an instance of GaussianMixture, not {type(gmm_model)}")
        super().__init__(feature_extractor,
                         gmm_model,
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
            num_descriptors = len(descriptors)

            mixture_weights = self.clustering_model.weights_
            means = self.clustering_model.means_
            covariances = self.clustering_model.covariances_

            posterior_probabilities = self.clustering_model.predict_proba(descriptors)

            # Statistics necessary to compute GMM gradients wrt its parameters
            pp_sum = posterior_probabilities.mean(axis=0, keepdims=True).T
            pp_x = posterior_probabilities.T.dot(descriptors) / num_descriptors
            pp_x_2 = posterior_probabilities.T.dot(np.power(descriptors, 2)) / num_descriptors

            # Compute GMM gradients wrt its parameters
            d_pi = pp_sum.squeeze() - mixture_weights

            d_mu = pp_x - pp_sum * means

            d_sigma_t1 = pp_sum * np.power(means, 2)
            d_sigma_t2 = pp_sum * covariances
            d_sigma_t3 = 2 * pp_x * means
            d_sigma = -pp_x_2 - d_sigma_t1 + d_sigma_t2 + d_sigma_t3

            # Apply analytical diagonal normalization
            sqrt_mixture_weights = np.sqrt(mixture_weights)
            d_pi /= sqrt_mixture_weights
            d_mu /= sqrt_mixture_weights[:, np.newaxis] * np.sqrt(covariances)
            d_sigma /= np.sqrt(2) * sqrt_mixture_weights[:, np.newaxis] * covariances

            # Concatenate GMM gradients to form Fisher vector representation
            descriptor_vector = np.hstack((d_pi, d_mu.ravel(), d_sigma.ravel()))
            descriptor_vector = descriptor_vector.reshape(1, -1)

            # Power normalization and L2 normalization
            descriptor_vector = np.sign(descriptor_vector) * np.power(np.abs(descriptor_vector), self.power_norm_weight)
            norm = np.linalg.norm(descriptor_vector, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
            descriptor_vector = descriptor_vector / norm

            if self.flatten:
                descriptor_vector = descriptor_vector.flatten()
            all_encodings.append(descriptor_vector)

        return np.vstack(all_encodings)
