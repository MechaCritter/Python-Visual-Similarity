from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ..features import RootSIFT
from ..features._features import FeatureExtractorBase
from ..encoders._base_encoder import ImageEncoderBase
from .._utils import cosine_similarity


class FisherVectorEncoder(ImageEncoderBase):
    """
    This class serves as an encoder that transforms input images into Fisher Vector descriptors.

    The Fisher Vector representation is based on the gradients of the GMM parameters
    (weights, means, and covariances) with respect to the feature descriptors extracted
    from the images. The representation is optionally power-normalized and L2-normalized.

    The output when calling `compute_vector` has shape (2 * num_clusters * feature_dim + num_clusters,).

    :param feature_extractor: Feature extractor instance. Default is RootSIFT
    :param gmm_model: Gaussian Mixture Model instance from scikit-learn.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed encoding vector (default: True).
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
    :param pca: PCA model for dimensionality reduction (optional).
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.

    References:
    ==========
    [1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """
    def __init__(self,
                 feature_extractor: FeatureExtractorBase=RootSIFT(),
                    weights=None,
                 gmm_model: GaussianMixture=None,
                 power_norm_weight: float = 0.5,
                 norm_order: int = 2,
                 epsilon: float = 1e-9,
                 flatten: bool = True,
                 similarity_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
                 pca: PCA = None,
                 raise_error_when_pca_incompatible: bool = False):
        if gmm_model is not None:
            if not isinstance(gmm_model, GaussianMixture):
                raise ValueError(f"The clustering model must be an instance of GaussianMixture, not {type(gmm_model)}")
        if weights is not None:
            if (weights_class:=weights.__class__.__name__) != 'GMMWeights':
                raise ValueError(f"You can only pass an instance of GMMWeights, not {weights_class}")
        super().__init__(feature_extractor,
                            weights,
                         gmm_model,
                         similarity_func,
                         power_norm_weight,
                         norm_order,
                         epsilon,
                         flatten,
                         pca,
                         raise_error_when_pca_incompatible)

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
