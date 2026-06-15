from typing import Any, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from .._utils import cosine_similarity
from ..clustering import PCA, ClusteringModelBase, GaussianMixtureModel
from ..encoders._base_encoder import GMMWeights, ImageEncoderBase
from ..typing import (
    Float64NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
)
from .utils import iter_images


class FisherVectorEncoder(ImageEncoderBase):
    """
    This class serves as an encoder that transforms input images into Fisher Vector descriptors.

    The Fisher Vector representation is based on the gradients of the GMM parameters
    (weights, means, and covariances) with respect to the feature descriptors extracted
    from the images. The representation is optionally power-normalized and L2-normalized.

    The Gaussian Mixture Model is configured from the parameters passed to
    this constructor (``n_components`` plus the optional ``gmm_params``
    dictionary) and fitted by calling :meth:`learn`. An optional PCA
    model for dimensionality reduction is configured the same way via
    ``pca_params``.

    The output when calling `encode` has shape (2 * num_clusters * feature_dim + num_clusters,).

    :param feature_extractor: Feature extractor instance. Default is RootSIFT
    :param weights: Pretrained GMM weights to load (deprecated).
    :param n_components: Number of Gaussian mixture components (visual words) to use.
    :param gmm_params: Dictionary of additional keyword arguments forwarded
        verbatim to :class:`sklearn.mixture.GaussianMixture` (e.g. ``{"random_state": 0}``).
    :param pca_params: Dictionary of keyword arguments for the PCA model used
        for dimensionality reduction (see ``pyvisim.clustering.PCA``); must
        include ``n_components``. If omitted, no PCA is applied.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed encoding vector (default: True).
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.

    References:
    ==========
    [1] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """

    _clustering_model_cls = GaussianMixtureModel

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        weights: GMMWeights | None = None,
        n_components: int = 256,
        gmm_params: dict[str, Any] | None = None,
        pca_params: dict[str, Any] | None = None,
        power_norm_weight: float = 0.5,
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        similarity_func: SimilarityFunc = cosine_similarity,
        raise_error_when_pca_incompatible: bool = False,
    ):
        if weights is not None:
            if (weights_class := weights.__class__.__name__) != "GMMWeights":
                raise ValueError(
                    f"You can only pass an instance of GMMWeights, not {weights_class}"
                )
        if gmm_params and "n_components" in gmm_params:
            raise ValueError(
                "Pass 'n_components' directly to FisherVectorEncoder instead of inside gmm_params."
            )
        clustering_model = (
            GaussianMixtureModel(n_components=n_components, **(gmm_params or {}))
            if weights is None
            else None
        )
        pca = PCA(**pca_params) if weights is None and pca_params is not None else None
        super().__init__(
            feature_extractor=feature_extractor,
            clustering_model=clustering_model,
            weights=weights,
            similarity_func=similarity_func,
            power_norm_weight=power_norm_weight,
            norm_order=norm_order,
            epsilon=epsilon,
            flatten=flatten,
            pca=pca,
            raise_error_when_pca_incompatible=raise_error_when_pca_incompatible,
        )

    @property
    def clustering_model(self) -> GaussianMixtureModel:
        return cast(GaussianMixtureModel, self._clustering_model)

    @clustering_model.setter
    def clustering_model(self, model: ClusteringModelBase) -> None:
        if not isinstance(model, GaussianMixtureModel):
            raise ValueError(
                f"The clustering model must be an instance of pyvisim.clustering.GaussianMixtureModel, not {type(model)}"
            )
        self._set_clustering_model(model)

    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float64NumpyArray:
        """
        Encode one or more images into Fisher Vector descriptors.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: ``(N, 2 × n_components × feature_dim + n_components)`` array of
            Fisher Vector encodings.
        """
        all_encodings = []
        for image in iter_images(images, dims=dims, value_range=value_range):
            descriptors: FloatNumpyArray = self.feature_extractor(image)
            if self.pca:
                descriptors = self.pca.transform(descriptors.astype(np.float32))
            num_descriptors = len(descriptors)

            mixture_weights = self.clustering_model.weights
            means = self.clustering_model.means
            covariances = self.clustering_model.covariances

            posterior_probabilities = self.clustering_model.predict_proba(descriptors)

            # Statistics necessary to compute GMM gradients wrt its parameters
            pp_sum = posterior_probabilities.mean(axis=0, keepdims=True).T
            pp_x = posterior_probabilities.T.dot(descriptors) / num_descriptors
            pp_x_2 = (
                posterior_probabilities.T.dot(np.power(descriptors, 2))
                / num_descriptors
            )

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
            descriptor_vector = np.sign(descriptor_vector) * np.power(
                np.abs(descriptor_vector), self.power_norm_weight
            )
            norm = (
                np.linalg.norm(
                    descriptor_vector, axis=1, ord=self.norm_order, keepdims=True
                )
                + self.epsilon
            )
            descriptor_vector = descriptor_vector / norm

            if self.flatten:
                descriptor_vector = descriptor_vector.flatten()
            all_encodings.append(descriptor_vector)

        return np.vstack(all_encodings)
