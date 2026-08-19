import abc
import pathlib
import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np

from .._base_classes import FeatureExtractorBase, SimilarityMetric
from .._config import setup_logging
from .._errors import NotFittedError
from .._utils import get_similarity_func
from ..features._registry import feature_extractor_from_dict
from ..features._root_sift import RootSIFT
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
)
from ..utils.image_utils import iter_images
from ._clustering import PCA, ClusteringModelBase
from ._serialization import load_encoder_state, save_encoder_state

setup_logging()

_ENCODER_FILE_SUFFIX = ".encoder"
_CLUSERING_ENCODER_FILE_FORMAT_VERSION = 1
_CLUSERING_ENCODER_FILE_FORMAT_VERSION_COMPATIBILITY: dict[tuple[int, int], bool] = {
    # TODO: when the next _CLUSERING_ENCODER_FILE_FORMAT_VERSION comes, check if
    # it's forward / backward compatible, then add entries like:
    # (1, 2): True,  # version 1 can read files from version 2
    #
    # However, (2, 1) might not be True!!
}
_EncoderT = TypeVar("_EncoderT", bound="ImageEmbedderBase")
_ClusteringEncoderT = TypeVar("_ClusteringEncoderT", bound="ClusteringBasedEncoder")


class ImageEmbedderBase(SimilarityMetric):
    """
    Base class for all image embedders. An image embedder generates a vector
    representation of an image which can be used for indexing, retrieval,
    clustering or classification tasks.

    This base only knows how to embed images into vectors (:meth:`embed`) and
    how to compare those vectors with a similarity function. Subclasses add the
    machinery they need on top of that, e.g. a feature extractor
    (:class:`FeatureBasedEncoder`) or a clustering model
    (:class:`ClusteringBasedEncoder`).

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    """

    #: Keys a serialised state must contain to be a valid encoder file.
    #: Subclasses extend this with their own required keys.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"encoder_class", "similarity_func"}
    )

    def __init__(self, similarity_func: str = "cosine"):
        # Set important attributes via setters to trigger error handling
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self.similarity_func = similarity_func

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

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialises this encoder into a JSON-safe state dictionary.

        Every concrete encoder must define how it serialises itself. The
        returned mapping has to contain at least the keys listed in
        :attr:`_STATE_KEYS` (notably ``"encoder_class"``) so that
        :meth:`load_from_disk` can validate the file and dispatch it to the
        right class. Arrays may be embedded as ``__ndarray__`` nodes; the
        serialization layer stores them as binary tensors.

        :return: A JSON-safe encoder description suitable for :meth:`from_dict`.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dict(cls: type[_EncoderT], state: dict[str, Any]) -> _EncoderT:
        """
        Rebuilds an encoder from a dictionary produced by :meth:`to_dict`.

        :param state: A JSON-safe encoder description from :meth:`to_dict`.
        :return: A ready-to-use encoder instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embeds one or more images into a batch of vector representations.

        Each image is normalized to a canonical ``uint8`` ``(H, W, C)`` array
        before feature extraction, so NumPy arrays, torch tensors and other
        array-like inputs are all accepted. When a batch axis is present (via
        ``dims``), every image in the batch is embedded.

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
        Computes vector embeddings for two images and calculates the similarity score between them.

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
        vector1 = self.embed(images1, dims=dims, value_range=value_range)
        vector2 = self.embed(images2, dims=dims, value_range=value_range)
        result = self.similarity_func(vector1, vector2)
        return np.asarray(result, dtype=np.float32)

    def save_to_disk(self, path: str | pathlib.Path) -> pathlib.Path:
        """
        Saves the serialised state of this encoder to a ``.encoder`` file.

        :param path: Target file path. The ``.encoder`` suffix is appended if missing.
        :return: The path of the written file.
        :raises NotFittedError: If the encoder is not ready to be serialised
            (see :meth:`to_dict`).
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
        if not cls._STATE_KEYS.issubset(state):
            raise ValueError(f"File {path} is not a valid .encoder file.")
        # TODO: in the future, verify the file's format version against the
        # class-specific compatibility table before reconstructing.
        if state["encoder_class"] != cls.__name__:
            raise ValueError(
                f"File {path} was saved by {state['encoder_class']}. "
                f"Load it with {state['encoder_class']}.load_from_disk instead."
            )
        return cls.from_dict(state)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(similarity_func={self.similarity_func_name})"
        )


class FeatureBasedEncoder(ImageEmbedderBase):
    """
    Base class for encoders that derive their vector representation from local
    features extracted by a :class:`~pyvisim.features.FeatureExtractorBase`
    (e.g. SIFT, SURF or deep features).

    :param feature_extractor: Feature extractor instance (should implement
        ``__call__``). Defaults to :class:`~pyvisim.features.RootSIFT`.
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        similarity_func: str = "cosine",
    ):
        self._feature_extractor: FeatureExtractorBase
        super().__init__(similarity_func=similarity_func)
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else RootSIFT()
        )

    @property
    def feature_extractor(self) -> FeatureExtractorBase:
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, feature_extractor: FeatureExtractorBase) -> None:
        self._validate_feature_extractor(feature_extractor)
        self._feature_extractor = feature_extractor

    def _validate_feature_extractor(
        self, feature_extractor: FeatureExtractorBase
    ) -> None:
        """
        Validates a candidate feature extractor before it is stored.

        :param feature_extractor: Feature extractor to validate.
        :raises TypeError: If ``feature_extractor`` is not a
            :class:`~pyvisim.features.FeatureExtractorBase` instance.
        """
        if not isinstance(feature_extractor, FeatureExtractorBase):
            raise TypeError(
                f"feature_extractor must be an instance of FeatureExtractorBase, not {type(feature_extractor)}"
            )


class ClusteringBasedEncoder(FeatureBasedEncoder):
    """
    Base class for image encoders that aggregate local features with a
    clustering model. Subclasses (VLAD and Fisher Vector) use a combination of:

    - A feature extractor: Extract local features from an image (e.g. SIFT, SURF, or Deep Features).
    - A clustering model (K-Means for VLAD or GMM for Fisher Vector): aggregates local features to their
    nearest centroids to produce fix-sized vectors.
    - A similarity function: computes a single float value from the vector representations that represents
    the similarity between two images.

    The encoding can be used for indexing, retrieval, clustering or classification tasks.
    :param feature_extractor: Feature extractor instance (should implement __call__).
        Defaults to RootSIFT.
    :param clustering_model: Clustering model used for generating descriptors.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: Name of the built-in similarity metric to use. One of
    ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :param pca: PCA model for dimensionality reduction (optional). Subclasses build
    it from the ``pca_params`` dictionary passed to their constructors.
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, an Error will be raised"""

    _clustering_model_cls: ClassVar[type[ClusteringModelBase]]

    #: Keys that a serialised state must contain to be a valid encoder file.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
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

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase | None = None,
        clustering_model: ClusteringModelBase | None = None,
        similarity_func: str = "cosine",
        power_norm_weight: float = 1,
        norm_order: int = 2,
        epsilon: float = 1e-9,
        flatten: bool = True,
        pca: PCA | None = None,
        raise_error_when_pca_incompatible: bool = True,
    ):
        # Set important attributes via setters to trigger error handling
        self._clustering_model: ClusteringModelBase | None = None
        self._pca: PCA | None = None

        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.raise_error_when_pca_incompatible = raise_error_when_pca_incompatible

        # The feature extractor setter validates against the (currently unset)
        # PCA / clustering model, so both must already exist as ``None`` above.
        super().__init__(
            feature_extractor=feature_extractor, similarity_func=similarity_func
        )

        if pca is not None:
            self._set_pca(pca)
        if clustering_model is not None:
            self._set_clustering_model(clustering_model)

    def _validate_feature_extractor(
        self, feature_extractor: FeatureExtractorBase
    ) -> None:
        """
        Validates a candidate feature extractor against the configured PCA or
        clustering model before it is stored.

        In addition to the type check performed by the base class, the feature
        extractor output size must match the input size of the fitted PCA (if
        any) or otherwise the fitted clustering model.

        :param feature_extractor: Feature extractor to validate.
        :raises TypeError: If ``feature_extractor`` is not a
            :class:`~pyvisim.features.FeatureExtractorBase` instance.
        :raises RuntimeError: If its output size is incompatible with the
            fitted PCA or clustering model.
        """
        super()._validate_feature_extractor(feature_extractor)
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
        :raises ValueError: If ``pca`` is not a ``PCA`` instance or its dimensions are
            incompatible with the feature extractor or clustering model.
        """
        if not isinstance(pca, PCA):
            raise ValueError(
                f"The PCA model must be an instance of pyvisim.encoders._clustering.PCA, not {type(pca)}"
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

    def load_clustering_model_from_sklearn(self, model: Any) -> None:
        """
        Replaces this encoder's clustering model with one created from a
        scikit-learn estimator.

        The estimator's constructor arguments are translated by the
        clustering model class' ``from_sklearn`` method; if the estimator is
        fitted, its learned state is adopted and validated against the
        configured feature extractor / PCA. Only clustering model classes
        exposing ``from_sklearn`` support this — currently
        :class:`~pyvisim.encoders._clustering.KMeans` (VLAD) and
        :class:`~pyvisim.encoders._clustering.DiagCovarGaussianMixture`
        (Fisher Vector).

        :param model: A scikit-learn estimator of the type matching this
            encoder's clustering model (e.g. :class:`sklearn.cluster.KMeans`
            for VLAD), fitted or not.
        :raises NotImplementedError: If this encoder's clustering model class
            cannot be created from a scikit-learn estimator.
        :raises TypeError: If ``model`` is not of the supported estimator type.
        :raises RuntimeError: If the fitted estimator's input size is
            incompatible with the configured feature extractor or PCA.
        """
        from_sklearn = getattr(self._clustering_model_cls, "from_sklearn", None)
        if from_sklearn is None:
            raise NotImplementedError(
                f"{self._clustering_model_cls.__name__} cannot be created from "
                "a scikit-learn estimator."
            )
        self._set_clustering_model(from_sklearn(model))

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
            "format_version": _CLUSERING_ENCODER_FILE_FORMAT_VERSION,
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
    def from_dict(
        cls: type[_ClusteringEncoderT], state: dict[str, Any]
    ) -> _ClusteringEncoderT:
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
