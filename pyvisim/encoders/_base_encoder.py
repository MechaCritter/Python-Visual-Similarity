import abc
import pathlib
import warnings
from enum import Enum
from typing import Any, ClassVar, TypeVar

import numpy as np
from sklearn.exceptions import NotFittedError

from .._base_classes import FeatureExtractorBase, SimilarityMetric
from .._config import MODEL_FILES_PATH, setup_logging
from .._utils import get_similarity_func
from ..clustering import PCA, ClusteringModelBase
from ..features._registry import feature_extractor_from_dict
from ..features._root_sift import RootSIFT
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
)
from ._serialization import load_encoder_state, save_encoder_state
from .utils import iter_images

setup_logging()

_ENCODER_FILE_SUFFIX = ".encoder"
_ENCODER_FILE_FORMAT_VERSION = 1
_ENCODER_FILE_FORMAT_VERSION_COMPATIBILITY: dict[tuple[int, int], bool] = {
    # TODO: when the next _ENCODER_FILE_FORMAT_VERSION comes, check if
    # it's forward / backward compatible, then add entries like:
    # (1, 2): True,  # version 1 can read files from version 2
    #
    # However, (2, 1) might not be True!!
}
_ENCODER_STATE_KEYS = frozenset(
    {
        "encoder_class",
        "clustering_model",
        "pca",
        "power_norm_weight",
        "norm_order",
        "epsilon",
        "flatten",
        "raise_error_when_pca_incompatible",
        "similarity_func",
        "feature_extractor",
    }
)


_EncoderT = TypeVar("_EncoderT", bound="ImageEncoderBase")


class _PretrainedModels(Enum):
    # TODO: removed with version 1.0.0
    @property
    def path(self) -> pathlib.Path:
        """Filesystem path of the bundled ``.encoder`` file."""
        return pathlib.Path(self.value)


class KMeansWeights(_PretrainedModels):
    """
    Pretrained K-Means weights trained on the Oxford-102 flower dataset.

    .. deprecated::
        Loading pretrained models via this enum is deprecated and will be
        removed in a future release. Use :meth:`VLADEncoder.from_pretrained`
        or ``load_from_disk`` with ``.encoder`` files instead.
    """

    # TODO: removed with version 1.0.0
    OXFORD102_K256_VGG16_PCA = (
        f"{MODEL_FILES_PATH}/vlad_oxford102_k256_vgg16_pca.encoder"
    )
    OXFORD102_K256_VGG16 = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_vgg16.encoder"
    OXFORD102_K256_ROOTSIFT_PCA = (
        f"{MODEL_FILES_PATH}/vlad_oxford102_k256_rootsift_pca.encoder"
    )
    OXFORD102_K256_ROOTSIFT = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_rootsift.encoder"
    OXFORD102_K256_SIFT_PCA = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_sift_pca.encoder"
    OXFORD102_K256_SIFT = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_sift.encoder"


class _PCA(_PretrainedModels):
    # TODO: removed with version 1.0.0
    OXFORD102_PCA256_VGG16 = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_vgg16_pca.encoder"
    OXFORD102_PCA256_ROOTSIFT = (
        f"{MODEL_FILES_PATH}/vlad_oxford102_k256_rootsift_pca.encoder"
    )
    OXFORD102_PCA256_SIFT = f"{MODEL_FILES_PATH}/vlad_oxford102_k256_sift_pca.encoder"


class GMMWeights(_PretrainedModels):
    """
    Pretrained GMM weights trained on the Oxford-102 flower dataset.

    .. deprecated::
        Loading pretrained models via this enum is deprecated and will be
        removed in a future release. Use
        :meth:`FisherVectorEncoder.from_pretrained` or ``load_from_disk``
        with ``.encoder`` files instead.
    """

    # TODO: removed with version 1.0.0
    OXFORD102_K256_VGG16_PCA = (
        f"{MODEL_FILES_PATH}/fisher_oxford102_k256_vgg16_pca.encoder"
    )
    OXFORD102_K256_VGG16 = f"{MODEL_FILES_PATH}/fisher_oxford102_k256_vgg16.encoder"
    OXFORD102_K256_ROOTSIFT_PCA = (
        f"{MODEL_FILES_PATH}/fisher_oxford102_k256_rootsift_pca.encoder"
    )
    OXFORD102_K256_ROOTSIFT = (
        f"{MODEL_FILES_PATH}/fisher_oxford102_k256_rootsift.encoder"
    )
    OXFORD102_K256_SIFT_PCA = (
        f"{MODEL_FILES_PATH}/fisher_oxford102_k256_sift_pca.encoder"
    )
    OXFORD102_K256_SIFT = f"{MODEL_FILES_PATH}/fisher_oxford102_k256_sift.encoder"


# TODO: removed with version 1.0.0
_CLUSTERING_TO_PCA_MAPPING = {
    KMeansWeights.OXFORD102_K256_VGG16_PCA: _PCA.OXFORD102_PCA256_VGG16,
    KMeansWeights.OXFORD102_K256_ROOTSIFT_PCA: _PCA.OXFORD102_PCA256_ROOTSIFT,
    KMeansWeights.OXFORD102_K256_SIFT_PCA: _PCA.OXFORD102_PCA256_SIFT,
    GMMWeights.OXFORD102_K256_VGG16_PCA: _PCA.OXFORD102_PCA256_VGG16,
    GMMWeights.OXFORD102_K256_ROOTSIFT_PCA: _PCA.OXFORD102_PCA256_ROOTSIFT,
    GMMWeights.OXFORD102_K256_SIFT_PCA: _PCA.OXFORD102_PCA256_SIFT,
}


class _PretrainedEncoder(Enum):
    """
    Base for pretrained-encoder enums backed by bundled ``.encoder`` files.

    Concrete members live next to their encoders:
    :class:`~pyvisim.encoders.PretrainedVLAD` (in ``vlad.py``) and
    :class:`~pyvisim.encoders.PretrainedFisher` (in ``fisher_vector.py``).
    """

    @property
    def path(self) -> pathlib.Path:
        """Filesystem path of the bundled ``.encoder`` file."""
        return pathlib.Path(self.value)


class ImageEncoderBase(SimilarityMetric):
    """
    Base class for image encoders. An image encoder is a class that
    generates a vector representation of an image. Subclasses use a combination of:

    - A feature extractor: Extract local features from an image (e.g. SIFT, SURF, or Deep Features).
    - A clustering model (K-Means for VLAD or GMM for Fisher Vector): aggregates local features to their
    nearest centroids to produce fix-sized vectors.
    - A similarity function: computes a single float value from the vector representations that represents
    the similarity between two images.

    The encoding can be used for indexing, retrieval, clustering or classification tasks.
    :param feature_extractor: Feature extractor instance (should implement __call__).
        Defaults to RootSIFT.
    :param clustering_model: Clustering model (see ``pyvisim.clustering``)
    used for generating descriptors.
    :param weights: Pretrained model for clustering. If provided, the clustering model will be loaded from the file,
    and `clustering_model` and `pca` parameters will be ignored.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
    ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param pca: PCA model (see ``pyvisim.clustering``) for dimensionality reduction
    (optional). Subclasses build it from the ``pca_params`` dictionary passed to
    their constructors.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, an Error will be raised"""

    _clustering_model_cls: ClassVar[type[ClusteringModelBase]]

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        clustering_model: ClusteringModelBase | None = None,
        weights: KMeansWeights
        | GMMWeights
        | None = None,  # TODO: removed with version 1.0.0
        similarity_func: str = "cosine",
        power_norm_weight: float = 1,
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        pca: PCA | None = None,
        raise_error_when_pca_incompatible: bool = True,
    ):
        # Set important attributes via setters to trigger error handling
        self._feature_extractor: FeatureExtractorBase
        self._clustering_model: ClusteringModelBase | None = None
        self._pca: PCA | None = None
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str

        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.raise_error_when_pca_incompatible = raise_error_when_pca_incompatible

        self.similarity_func = similarity_func
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else RootSIFT()
        )

        if weights is not None:
            self._load_pretrained_weights(weights)  # TODO: removed with version 1.0.0
        else:
            if pca is not None:
                self._set_pca(pca)
            if clustering_model is not None:
                self._set_clustering_model(clustering_model)

    # TODO: removed with version 1.0.0
    def _load_pretrained_weights(self, weights: KMeansWeights | GMMWeights) -> None:
        """
        Loads a pretrained clustering model (and its matching PCA, if any) from
        a bundled ``.encoder`` file into this encoder.

        .. deprecated::
            Will be removed in version 1.0.0 together with the weight enums.
            Use :meth:`from_pretrained` or :meth:`load_from_disk` instead.

        :param weights: Pretrained weight enum member to load.
        """
        warnings.warn(
            "Loading pretrained models via KMeansWeights/GMMWeights is "
            "deprecated and will be removed in a future release. Use "
            "from_pretrained()/load_from_disk() with .encoder files instead.",
            FutureWarning,
            stacklevel=3,
        )
        if "PCA" in weights.name:
            pca_state = load_encoder_state(_CLUSTERING_TO_PCA_MAPPING[weights].path)
            self._set_pca(PCA.from_dict(pca_state["pca"]))
        clustering_state = load_encoder_state(weights.path)
        self._set_clustering_model(
            self._clustering_model_cls.from_dict(clustering_state["clustering_model"])
        )

    @property
    def feature_extractor(self) -> FeatureExtractorBase:
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, feature_extractor: FeatureExtractorBase) -> None:
        if not isinstance(feature_extractor, FeatureExtractorBase):
            raise TypeError(
                f"feature_extractor must be an instance of FeatureExtractorBase, not {type(feature_extractor)}"
            )
        if self._pca is not None and self._pca.is_fitted:
            if feature_extractor.output_dim != self._pca.n_features_in:
                raise RuntimeError(
                    f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                    f"But PCA accepts input dim {self._pca.n_features_in}"
                )
        else:
            if self._clustering_model is not None and self._clustering_model.is_fitted:
                if feature_extractor.output_dim != self._clustering_model.n_features_in:
                    raise RuntimeError(
                        f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                        f"But clustering model accepts input dim {self._clustering_model.n_features_in}"
                    )
        self._feature_extractor = feature_extractor

    @property
    def similarity_func(self) -> SimilarityFunc:
        """The resolved similarity function callable."""
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, name: str) -> None:
        self._similarity_func = get_similarity_func(name)
        self._similarity_func_name = name

    @property
    def similarity_func_name(self) -> str:
        """The name of the configured similarity metric (e.g. ``"cosine"``)."""
        return self._similarity_func_name

    @property
    def clustering_model(self) -> ClusteringModelBase | None:
        return self._clustering_model

    def _set_clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        """
        Validates the given clustering model against the current PCA or
        feature extractor and stores it.

        Dimension checks only apply to fitted models; an unfitted model is
        stored as-is and validated once it is fitted via :meth:`learn`.

        :param clustering_model: Clustering model to validate and store.
        """
        if not clustering_model.is_fitted:
            self._clustering_model = clustering_model
            return
        if self._pca is not None and self._pca.is_fitted:
            if self._pca.n_components != clustering_model.n_features_in:
                if self.raise_error_when_pca_incompatible:
                    raise RuntimeError(
                        f"PCA is incompatible with the new clustering model. "
                        f"PCA input size: {self._pca.n_components}, "
                        f"New clustering model input size: {clustering_model.n_features_in}. "
                        f"If you want the PCA to be reset to None instead, set raise_error_when_pca_incompatible=False."
                    )
                warnings.warn(
                    f"PCA is incompatible with the new clustering model. "
                    f"PCA input size: {self._pca.n_components}, "
                    f"New clustering model input size: {clustering_model.n_features_in}. "
                    "PCA will be reset to None to avoid errors."
                    "If you want to raise an Error instead when this happens, set raise_error_when_pca_incompatible=False.",
                    stacklevel=2,
                )
                self._pca = None
        else:
            if self._feature_extractor.output_dim != clustering_model.n_features_in:
                raise RuntimeError(
                    "Feature extractor output size has to match the clustering model input size. "
                    f"Feature extractor has output size {self._feature_extractor.output_dim}, "
                    f"while clustering model has input size {clustering_model.n_features_in}"
                )
        self._clustering_model = clustering_model

    @property
    def pca(self) -> PCA | None:
        return self._pca

    def _set_pca(self, pca: PCA) -> None:
        """
        Validates and stores the given PCA model.

        :param pca: PCA model to validate and store.
        :raises ValueError: If ``pca`` is not a :class:`~pyvisim.clustering.PCA` instance
            or its dimensions are incompatible with the feature extractor or clustering model.
        """
        if not isinstance(pca, PCA):
            raise ValueError(
                f"The PCA model must be an instance of pyvisim.clustering.PCA, not {type(pca)}"
            )
        if not pca.is_fitted:
            self._pca = pca
            return
        if pca.n_features_in != self._feature_extractor.output_dim:
            raise ValueError(
                "PCA input size has to match the feature extractor output size. "
                f"PCA model has input size {pca.n_features_in}, "
                f"while feature extractor has output size {self._feature_extractor.output_dim}"
            )

        if self._clustering_model is not None and self._clustering_model.is_fitted:
            if pca.n_components != self._clustering_model.n_features_in:
                raise ValueError(
                    "PCA input size has to match the clustering model input size."
                    f"PCA model has input size {pca.n_components}, "
                    f"while clustering model has input size {self._clustering_model.n_features_in}"
                )

        self._pca = pca

    def learn(
        self,
        images: ImageInput,
        /,
        *,
        dim_reduction_factor: int | None = None,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        """
        Learns the visual vocabulary from the given images.

        The clustering model configured at initialization (with the
        scikit-learn parameters passed to the encoder constructor) is fitted
        on the extracted features. If a PCA model is configured, the features
        are reduced with it first (fitting it beforehand if necessary).

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Each image is normalized to a canonical
            ``uint8`` ``(H, W, C)`` array before feature extraction.
        :param dim_reduction_factor: If a value is provided, a new PCA model will be used to reduce the dimensionality of the feature space
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :raises RuntimeError: If the encoder has no clustering model configured.
        :raises ValueError: If dim_reduction_factor is provided but is not a positive integer.
        """
        if dim_reduction_factor is not None and (
            dim_reduction_factor <= 0 or not isinstance(dim_reduction_factor, int)
        ):
            raise ValueError("dim_reduction_factor must be a positive integer.")
        if self._clustering_model is None:
            raise RuntimeError(
                "This encoder has no clustering model to fit. "
                "Configure one via the constructor parameters."
            )
        features: FloatNumpyArray = np.vstack(
            [
                self.feature_extractor(image)
                for image in iter_images(images, dims=dims, value_range=value_range)
            ]
        )
        print("[INFO] Learning the visual vocabulary with the following parameters:")
        print("   - Number of clusters:", self._clustering_model.n_clusters)
        print("   - Feature Extractor used:", self.feature_extractor.__class__.__name__)
        print("   - Dimension of the feature space:", feat_dim := features.shape[1])
        if dim_reduction_factor:
            if (n_components := feat_dim // dim_reduction_factor) <= 0:
                raise ValueError(
                    f"dim_reduction_factor {dim_reduction_factor} is too large for the feature dimension {feat_dim}. "
                    f"Resulting PCA components would be {n_components}. Please choose a smaller dim_reduction_factor."
                )
            self._pca = PCA(n_components=n_components)
        if self._pca is not None:
            if not self._pca.is_fitted:
                self._pca.fit(features)
            features = self._pca.transform(features)
            print("   - New dimension after PCA reduction:", self._pca.n_components)
        self._clustering_model.fit(features)

    @classmethod
    def from_pretrained(
        cls: type[_EncoderT], pretrained: _PretrainedEncoder
    ) -> _EncoderT:
        """
        Loads a bundled pretrained encoder.

        :param pretrained: A pretrained-encoder enum member, e.g.
            :class:`PretrainedVLAD` for :class:`VLADEncoder` or
            :class:`PretrainedFisher` for :class:`FisherVectorEncoder`.
        :return: A ready-to-use encoder instance with its feature extractor and
            similarity metric restored from the file.
        :raises ValueError: If the chosen pretrained encoder was not saved by
            this encoder class (e.g. loading a Fisher encoder as VLAD).
        """
        return cls.load_from_disk(pretrained.path)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the learned encoder into a JSON-safe state dictionary.

        The returned mapping describes the encoder class, its clustering model,
        optional PCA, normalization hyperparameters, similarity metric and
        feature extractor. Arrays are kept as ``__ndarray__`` nodes so the
        whole dictionary can be embedded inside a larger state (e.g. an
        :class:`~pyvisim.image_store.InMemoryImageEmbeddingStore`).

        :return: A JSON-safe encoder description suitable for
            :meth:`from_dict`.
        :raises NotFittedError: If the clustering model is missing or not fitted.
        """
        if self._clustering_model is None or not self._clustering_model.is_fitted:
            raise NotFittedError(
                "Cannot serialise an encoder whose clustering model is not "
                "fitted. Call 'learn' first."
            )
        return {
            "format_version": _ENCODER_FILE_FORMAT_VERSION,
            "encoder_class": type(self).__name__,
            "clustering_model": self._clustering_model.to_dict(),
            "pca": self._pca.to_dict() if self._pca is not None else None,
            "power_norm_weight": self.power_norm_weight,
            "norm_order": self.norm_order,
            "epsilon": self.epsilon,
            "flatten": self.flatten,
            "raise_error_when_pca_incompatible": self.raise_error_when_pca_incompatible,
            "similarity_func": self._similarity_func_name,
            "feature_extractor": self._feature_extractor.to_dict(),
        }

    @classmethod
    def from_dict(cls: type[_EncoderT], state: dict[str, Any]) -> _EncoderT:
        """
        Rebuilds an encoder from a dictionary produced by :meth:`to_dict`.

        The caller is responsible for dispatching ``state["encoder_class"]`` to
        the matching encoder class; this method trusts that ``cls`` is correct.

        :param state: A JSON-safe encoder description from :meth:`to_dict`.
        :return: A ready-to-use encoder instance.
        """
        encoder = cls(
            feature_extractor=feature_extractor_from_dict(state["feature_extractor"]),
            similarity_func=state["similarity_func"],
            power_norm_weight=state["power_norm_weight"],
            norm_order=state["norm_order"],
            epsilon=state["epsilon"],
            flatten=state["flatten"],
            raise_error_when_pca_incompatible=state[
                "raise_error_when_pca_incompatible"
            ],
        )
        if state["pca"] is not None:
            encoder._set_pca(PCA.from_dict(state["pca"]))
        encoder._set_clustering_model(
            cls._clustering_model_cls.from_dict(state["clustering_model"])
        )
        return encoder

    def save_to_disk(self, path: str | pathlib.Path) -> pathlib.Path:
        """
        Saves the learned state of this encoder to a ``.encoder`` file.


        :param path: Target file path. The ``.encoder`` suffix is appended if missing.
        :return: The path of the written file.
        :raises NotFittedError: If the clustering model is missing or not fitted.
        """
        path = pathlib.Path(path)
        if path.suffix != _ENCODER_FILE_SUFFIX:
            path = path.with_name(path.name + _ENCODER_FILE_SUFFIX)
        save_encoder_state(self.to_dict(), path)
        return path

    @classmethod
    def load_from_disk(
        cls: type[_EncoderT],
        path: str | pathlib.Path,
    ) -> _EncoderT:
        """
        Loads an encoder previously saved with :meth:`save_to_disk`.

        :param path: Path to the ``.encoder`` file.
        :return: A ready-to-use encoder instance.
        :raises ValueError: If the file is not a valid ``.encoder`` file or
            was saved by a different encoder class.
        """
        state = load_encoder_state(pathlib.Path(path))
        if not _ENCODER_STATE_KEYS.issubset(state):
            raise ValueError(f"File {path} is not a valid .encoder file.")
        # TODO: in the future, verify format version by checking
        # compatibility via _ENCODER_FILE_FORMAT_VERSION_COMPATIBILITY
        if state["encoder_class"] != cls.__name__:
            raise ValueError(
                f"File {path} was saved by {state['encoder_class']}. "
                f"Load it with {state['encoder_class']}.load_from_disk instead."
            )
        return cls.from_dict(state)

    @abc.abstractmethod
    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Encodes one or more images into a batch of vector representations.

        Each image is normalized to a canonical ``uint8`` ``(H, W, C)`` array
        before feature extraction, so NumPy arrays, torch tensors and other
        array-like inputs are all accepted. When a batch axis is present (via
        ``dims``), every image in the batch is encoded.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Consider using an iterator for large datasets.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: vector representations of the given images
        """
        raise NotImplementedError

    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Computes vector encodings for two images and calculates the similarity score between them.

        :param images1: First (batch of) image(s) as ``MatLike``.
        :param images2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: Similarity matrix between the two image batches.
        """
        vector1 = self.encode(images1, dims=dims, value_range=value_range)
        vector2 = self.encode(images2, dims=dims, value_range=value_range)
        result = self.similarity_func(vector1, vector2)
        return np.asarray(result, dtype=np.float32)

    def __repr__(self) -> str:
        n_clusters = (
            self._clustering_model.n_clusters if self._clustering_model else None
        )
        return (
            self.__class__.__name__
            + f"(feature_extractor={self.feature_extractor.__class__.__name__}, \n"
            f"similarity_func={self.similarity_func_name}, \n"
            f"Number of cluster={n_clusters}, \n"
            f"Power Norm Weight={self.power_norm_weight}, \n"
            f"Norm Order={self.norm_order})"
        )
