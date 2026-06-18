from typing import Any, cast

import numpy as np

from .._base_classes import FeatureExtractorBase
from .._config import MODEL_FILES_PATH
from ..clustering import PCA, ClusteringModelBase, KMeans
from ..encoders._base_encoder import (
    ImageEncoderBase,
    KMeansWeights,
    _PretrainedEncoder,
)
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
)
from .utils import iter_images


class PretrainedVLAD(_PretrainedEncoder):
    """
    Bundled pretrained VLAD encoders trained on the Oxford-102 flower dataset
    with ``k=256`` clusters. Load one with :meth:`VLADEncoder.from_pretrained`.

    Variants differ by the feature extractor (RootSIFT, SIFT or VGG16 deep
    features) and whether PCA dimensionality reduction was applied.
    """

    OXFORD102_K256_ROOTSIFT = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_rootsift.encoder"
    OXFORD102_K256_ROOTSIFT_PCA = (
        f"{MODEL_FILES_PATH}/vlad_oxford102_k256_rootsift_pca.encoder"
    )
    OXFORD102_K256_SIFT = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_sift.encoder"
    OXFORD102_K256_SIFT_PCA = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_sift_pca.encoder"
    OXFORD102_K256_VGG16 = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_vgg16.encoder"
    OXFORD102_K256_VGG16_PCA = (
        f"{MODEL_FILES_PATH}/vlad_oxford102_k256_vgg16_pca.encoder"
    )


class VLADEncoder(ImageEncoderBase):
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
    :param weights: Pretrained K-Means weights to load (deprecated).
    :param n_clusters: Number of K-Means clusters (visual words) to use.
    :param kmeans_params: Dictionary of additional keyword arguments forwarded
        verbatim to :class:`sklearn.cluster.KMeans` (e.g. ``{"random_state": 0}``).
    :param pca_params: Dictionary of keyword arguments for the PCA model used
        for dimensionality reduction (see ``pyvisim.clustering.PCA``); must
        include ``n_components``. If omitted, no PCA is applied.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, the PCA model will be reset to None.

    References:
    ==========
    [1] Relja Arandjelović and Andrew Zisserman, 'All About VLAD', Department of Engineering Science, University of Oxford.
    [2] Relja Arandjelović and Andrew Zisserman, "Three things everyone should know to improve object retrieval," Department of Engineering Science, University of Oxford.
    [3] Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick Pérez, and Cordelia Schmid, "Aggregating Local Image Descriptors into Compact Codes," IEEE.
    """

    _clustering_model_cls = KMeans

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        weights: KMeansWeights | None = None,  # TODO: removed with version 1.0.0
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
        if weights is not None:
            if (weights_class := weights.__class__.__name__) != "KMeansWeights":
                raise ValueError(
                    f"You can only pass an instance of KMeansWeights, not {weights_class}"
                )
        if kmeans_params and "n_clusters" in kmeans_params:
            raise ValueError(
                "Pass 'n_clusters' directly to VLADEncoder instead of inside kmeans_params."
            )
        clustering_model = (
            KMeans(n_clusters=n_clusters, **(kmeans_params or {}))
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
    def clustering_model(self) -> KMeans:
        return cast(KMeans, self._clustering_model)

    @clustering_model.setter
    def clustering_model(self, model: ClusteringModelBase) -> None:
        if not isinstance(model, KMeans):
            raise ValueError(
                f"The clustering model must be an instance of pyvisim.clustering.KMeans, not {type(model)}"
            )
        self._set_clustering_model(model)

    def encode(
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
