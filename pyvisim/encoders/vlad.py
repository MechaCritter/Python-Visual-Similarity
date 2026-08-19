from typing import Any, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from ..encoders._base_encoder import ClusteringBasedEncoder
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
)
from ..utils.image_utils import iter_images
from ._clustering import PCA, ClusteringModelBase, KMeans


class VLADEncoder(ClusteringBasedEncoder):
    """
    This class encodes images into VLAD descriptor vectors
    using a chosen feature extractor and a K-Means clustering model,
    then compares two VLAD descriptor vectors with a user-specified
    or default (cosine) similarity function.

    The K-Means model is configured from the parameters passed to this
    constructor (``n_clusters`` plus the optional ``kmeans_params``
    dictionary) and fitted by calling :meth:`learn`. An optional PCA
    model for dimensionality reduction is configured the same way via
    ``pca_params``.

    The output when calling `encode` has shape (num_clusters * feature_dim,).

    You can use euclidean distance, manhattan distance, etc. as the similarity function.

    The encoding can be used for indexing, retrieval, clustering or classification tasks.

    :param feature_extractor: Feature extractor instance (should implement __call__).
        Defaults to RootSIFT.
    :param n_clusters: Number of K-Means clusters (visual words) to use.
    :param kmeans_params: Arguments for K-Means during vocabulary learning. See
        ``https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/encoders/vlad.md#k-means-parameters``.
    :param pca_params: Arguments for the Principal Component Analysis during vocabulary learning. See
        ``https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/encoders/pca.md#parameters``.
    :param power_norm_weight: Exponent for power normalization
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.

    .. rubric:: References

    - [1] Relja Arandjelović and Andrew Zisserman, 'All About VLAD', Department of Engineering Science, University of Oxford.
    - [2] Relja Arandjelović and Andrew Zisserman, "Three things everyone should know to improve object retrieval," Department of Engineering Science, University of Oxford.
    - [3] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """

    _clustering_model_cls = KMeans

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        n_clusters: int = 256,
        kmeans_params: dict[str, Any] | None = None,
        pca_params: dict[str, Any] | None = None,
        power_norm_weight: float = 1,  # no paper found where power norm weight is used for VLAD
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        similarity_func: str = "cosine",
        raise_error_when_pca_incompatible: bool = False,
    ) -> None:
        if kmeans_params and "n_clusters" in kmeans_params:
            raise ValueError(
                "Pass 'n_clusters' directly to VLADEncoder instead of inside kmeans_params."
            )
        clustering_model = KMeans(n_clusters=n_clusters, **(kmeans_params or {}))
        pca = PCA(**pca_params) if pca_params is not None else None
        super().__init__(
            feature_extractor=feature_extractor,
            clustering_model=clustering_model,
            similarity_func=similarity_func,
            power_norm_weight=power_norm_weight,
            norm_order=norm_order,
            epsilon=epsilon,
            flatten=flatten,
            pca=pca,
            raise_error_when_pca_incompatible=raise_error_when_pca_incompatible,
        )

    @property
    def clustering_model(self) -> KMeans:
        return cast(KMeans, self._clustering_model)

    def _set_clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        if not isinstance(clustering_model, KMeans):
            raise ValueError(
                f"The clustering model must be an instance of pyvisim.encoders._clustering.KMeans, not {type(clustering_model)}"
            )
        super()._set_clustering_model(clustering_model)

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Encode one or more images into VLAD descriptor vectors.

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
        :return: ``(N, n_clusters × feature_dim)`` array of VLAD encodings.
        """
        all_encodings = []
        for image in iter_images(images, dims=dims, value_range=value_range):
            descriptors: FloatNumpyArray = self.feature_extractor(image)
            if self.pca:
                descriptors = self.pca.transform(descriptors.astype(np.float32))

            if descriptors is None or descriptors.shape[0] == 0:
                raise ValueError(
                    "No descriptors found in the image. Cannot compute VLAD encoding."
                )

            labels = self.clustering_model.predict(descriptors.astype(np.float32))
            centroids = self.clustering_model.cluster_centers

            k = len(centroids)
            dim = descriptors.shape[1]
            descriptor_vector: FloatNumpyArray = np.zeros((k, dim), dtype=np.float32)

            for i, desc in enumerate(descriptors):
                cluster_id = labels[i]
                descriptor_vector[cluster_id] += desc - centroids[cluster_id]

            descriptor_vector = (
                np.sign(descriptor_vector)
                * np.abs(descriptor_vector) ** self.power_norm_weight
            )
            norms = (
                np.linalg.norm(
                    descriptor_vector, axis=1, ord=self.norm_order, keepdims=True
                )
                + self.epsilon
            )
            descriptor_vector = descriptor_vector / norms

            if self.flatten:
                descriptor_vector = descriptor_vector.flatten()

            all_encodings.append(descriptor_vector)

        return np.vstack(all_encodings)
