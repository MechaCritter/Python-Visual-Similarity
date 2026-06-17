import abc
import pathlib
import warnings
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from enum import Enum
from functools import wraps
from typing import Any, ClassVar, TypeVar, cast

import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

from .._base_classes import FeatureExtractorBase, SimilarityMetric
from .._config import PICKLE_MODEL_FILES_PATH, setup_logging
from .._utils import cosine_similarity
from ..clustering import PCA, ClusteringModelBase
from ..features._features import RootSIFT
from ..image_store import ImageEncodingMap
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
)
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
    }
)


# Helper Functions
def check_desired_output(
    similarity_func: Callable[[FloatNumpyArray, FloatNumpyArray], Any],
    vecs1: FloatNumpyArray,
    vecs2: FloatNumpyArray,
) -> SimilarityFunc:
    """
    Checks the output of the given similarity_func(vecs1, vecs2).
    Requirements:
    1) Output must be a NumPy array
    2) Output shape must be (len(vecs1), len(vecs2)) if batch
       or (1,1) if single
    3) If it fails, we degrade to a fallback method that
       loops over each row in vecs1 vs each row in vecs2.

    :param similarity_func: function that tries to compute similarities
                           between two arrays of shape (N, D) and (M, D).
    :param vecs1: (N, D) or (D,) array
    :param vecs2: (M, D) or (D,) array
    :return: A potentially wrapped function that always returns
             shape (N, M) as a NumPy array of floats
    """
    try:
        out = similarity_func(vecs1, vecs2)
    except Exception as e:
        warnings.warn(
            f"Similarity function threw an error: {e}. Falling back to row-wise loop.",
            stacklevel=2,
        )
        return _make_fallback_func(similarity_func)

    if not isinstance(out, np.ndarray):
        warnings.warn(
            f"Expected a NumPy array, got {type(out)}. Using fallback method.",
            stacklevel=2,
        )
        return _make_fallback_func(similarity_func)

    # Check shape
    # If vecs1 is shape (N, D) and vecs2 is shape (M, D), we expect out.shape = (N, M).
    # If single vector, it might produce shape (1,1) or just a float
    shape_ok = True
    if out.ndim == 2:
        if out.shape[0] != vecs1.shape[0] or out.shape[1] != vecs2.shape[0]:
            shape_ok = False
    elif out.ndim == 1 and out.size != 1:
        shape_ok = False

    if not shape_ok:
        warnings.warn(
            f"Output shape {out.shape} is not the expected (N, M). Expected output shape to be "
            f"({vecs1.shape[0]}, {vecs2.shape[0]}). Using fallback.",
            stacklevel=2,
        )
        return _make_fallback_func(similarity_func)

    return similarity_func


def _make_fallback_func(
    sim_func: Callable[[FloatNumpyArray, FloatNumpyArray], Any],
) -> SimilarityFunc:
    """
    Returns a new function that loops row-by-row if the original
    similarity function can't handle batch mode.
    """

    def fallback(vecs1: FloatNumpyArray, vecs2: FloatNumpyArray) -> Float32NumpyArray:
        N = vecs1.shape[0]  # (N, D)
        M = vecs2.shape[0]  # (M, D)
        out = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                out[i, j] = sim_func(vecs1[i : i + 1], vecs2[j : j + 1])
        return out

    try:
        return fallback
    except Exception as e:
        raise RuntimeError(
            f"Row-wise operation was not possible with the given similarity function: {e}"
            "Your function is invalid."
        ) from e


MethodT = TypeVar("MethodT", bound=Callable[..., Any])
_EncoderT = TypeVar("_EncoderT", bound="ImageEncoderBase")


def _tupleize_first_arg(func: MethodT) -> MethodT:  # noqa: UP047
    """
    # TODO: currently, the param 'image_paths' param is hardcoded. This should be more general
    # to be able to handle any variable name
    Pass this to the "encode" and "generate_encoding_map" methods to
    convert the input to a tuple so that it can be hashed by the lru_cache.
    """

    @wraps(func)
    def wrapper(self: Any, image_paths: Any, /, *args: Any, **kwargs: Any) -> Any:
        if isinstance(image_paths, (Iterator, MutableSequence)):
            image_paths = tuple(image_paths)
        return func(self, image_paths, *args, **kwargs)

    return cast(MethodT, wrapper)


class _PretrainedModels(Enum):
    def load(self) -> object:
        """Loads the model from the file path"""
        with open(self.value, "rb") as f:
            return joblib.load(f)


class KMeansWeights(_PretrainedModels):
    """
    Pretrained K-Means weights trained on the Oxford-102 flower dataset.

    .. deprecated::
        Loading pretrained models via this enum is deprecated and will be
        removed in a future release. Use ``save_to_disk``/``load_from_disk``
        with ``.encoder`` files instead.
    """

    OXFORD102_K256_VGG16_PCA = (
        f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_deep_features_vgg16_pca.pkl"
    )
    OXFORD102_K256_VGG16 = (
        f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_deep_features_vgg16_no_pca.pkl"
    )
    OXFORD102_K256_ROOTSIFT_PCA = (
        f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_root_sift_pca.pkl"
    )
    OXFORD102_K256_ROOTSIFT = (
        f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_root_sift_no_pca.pkl"
    )
    OXFORD102_K256_SIFT_PCA = f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_sift_pca.pkl"
    OXFORD102_K256_SIFT = f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_sift_no_pca.pkl"


class _PCA(_PretrainedModels):
    OXFORD102_PCA256_VGG16 = (
        f"{PICKLE_MODEL_FILES_PATH}/pca_k256_deep_features_vgg16_f2.pkl"
    )
    OXFORD102_PCA256_ROOTSIFT = f"{PICKLE_MODEL_FILES_PATH}/pca_k256_root_sift_f2.pkl"
    OXFORD102_PCA256_SIFT = f"{PICKLE_MODEL_FILES_PATH}/pca_k256_sift_f2.pkl"


class GMMWeights(_PretrainedModels):
    """
    Pretrained GMM weights trained on the Oxford-102 flower dataset.

    .. deprecated::
        Loading pretrained models via this enum is deprecated and will be
        removed in a future release. Use ``save_to_disk``/``load_from_disk``
        with ``.encoder`` files instead.
    """

    OXFORD102_K256_VGG16_PCA = (
        f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_deep_features_vgg16_pca.pkl"
    )
    OXFORD102_K256_VGG16 = (
        f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_deep_features_vgg16_no_pca.pkl"
    )
    OXFORD102_K256_ROOTSIFT_PCA = (
        f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_root_sift_pca.pkl"
    )
    OXFORD102_K256_ROOTSIFT = f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_root_sift_no_pca.pkl"
    OXFORD102_K256_SIFT_PCA = f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_sift_pca.pkl"
    OXFORD102_K256_SIFT = f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_sift_no_pca.pkl"


_CLUSTERING_TO_PCA_MAPPING = {
    KMeansWeights.OXFORD102_K256_VGG16_PCA: _PCA.OXFORD102_PCA256_VGG16,
    KMeansWeights.OXFORD102_K256_ROOTSIFT_PCA: _PCA.OXFORD102_PCA256_ROOTSIFT,
    KMeansWeights.OXFORD102_K256_SIFT_PCA: _PCA.OXFORD102_PCA256_SIFT,
    GMMWeights.OXFORD102_K256_VGG16_PCA: _PCA.OXFORD102_PCA256_VGG16,
    GMMWeights.OXFORD102_K256_ROOTSIFT_PCA: _PCA.OXFORD102_PCA256_ROOTSIFT,
    GMMWeights.OXFORD102_K256_SIFT_PCA: _PCA.OXFORD102_PCA256_SIFT,
}


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
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
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
        weights: KMeansWeights | GMMWeights | None = None,
        similarity_func: SimilarityFunc = cosine_similarity,
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
            self._load_pretrained_weights(weights)
        else:
            if pca is not None:
                self.pca = pca
            if clustering_model is not None:
                self.clustering_model = clustering_model

    def _load_pretrained_weights(self, weights: KMeansWeights | GMMWeights) -> None:
        """
        Loads a pretrained scikit-learn estimator (and its matching PCA,
        if any) into the encoder's clustering model class.

        .. deprecated::
            Will be removed in a future release together with the weight enums.

        :param weights: Pretrained weight enum member to load.
        """
        warnings.warn(
            "Loading pretrained models via KMeansWeights/GMMWeights is "
            "deprecated and will be removed in a future release. Use "
            "save_to_disk()/load_from_disk() with .encoder files instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if "PCA" in weights.name:
            self.pca = PCA._from_sklearn(_CLUSTERING_TO_PCA_MAPPING[weights].load())
        self.clustering_model = self._clustering_model_cls._from_sklearn(weights.load())

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
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, func: SimilarityFunc) -> None:
        dummy1, dummy2 = np.random.rand(10, 10), np.random.rand(10, 10)
        self._similarity_func = check_desired_output(func, dummy1, dummy2)

    @property
    def clustering_model(self) -> ClusteringModelBase | None:
        return self._clustering_model

    @clustering_model.setter
    def clustering_model(self, clustering_model: ClusteringModelBase) -> None:
        self._set_clustering_model(clustering_model)

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

    @pca.setter
    def pca(self, pca: PCA) -> None:
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

    def save_to_disk(self, path: str | pathlib.Path) -> pathlib.Path:
        """
        Saves the learned state of this encoder to a ``.encoder`` file.

        The file contains the fitted clustering model, the PCA model (if any)
        and the normalization hyperparameters. The feature extractor and the
        similarity function are not serialized; provide them again when
        calling :meth:`load_from_disk`.

        :param path: Target file path. The ``.encoder`` suffix is appended if missing.
        :return: The path of the written file.
        :raises NotFittedError: If the clustering model is missing or not fitted.
        """
        if self._clustering_model is None or not self._clustering_model.is_fitted:
            raise NotFittedError(
                "Cannot save an encoder whose clustering model is not fitted. "
                "Call 'learn' first."
            )
        path = pathlib.Path(path)
        if path.suffix != _ENCODER_FILE_SUFFIX:
            path = path.with_name(path.name + _ENCODER_FILE_SUFFIX)
        state = {
            "format_version": _ENCODER_FILE_FORMAT_VERSION,
            "encoder_class": type(self).__name__,
            "clustering_model": self._clustering_model,
            "pca": self._pca,
            "power_norm_weight": self.power_norm_weight,
            "norm_order": self.norm_order,
            "epsilon": self.epsilon,
            "flatten": self.flatten,
            "raise_error_when_pca_incompatible": self.raise_error_when_pca_incompatible,
        }
        joblib.dump(state, path)
        return path

    @classmethod
    def load_from_disk(
        cls: type[_EncoderT],
        path: str | pathlib.Path,
        *,
        feature_extractor: FeatureExtractorBase | None = None,
        similarity_func: SimilarityFunc = cosine_similarity,
    ) -> _EncoderT:
        """
        Loads an encoder previously saved with :meth:`save_to_disk`.

        :param path: Path to the ``.encoder`` file.
        :param feature_extractor: Feature extractor to use with the loaded
            encoder. Defaults to RootSIFT. Its output dimension has to match
            the input dimension of the saved PCA or clustering model.
        :param similarity_func: Similarity function to use with the loaded encoder.
        :return: A ready-to-use encoder instance.
        :raises ValueError: If the file is not a valid ``.encoder`` file or
            was saved by a different encoder class.
        """
        state = joblib.load(path)
        if not isinstance(state, dict) or not _ENCODER_STATE_KEYS.issubset(state):
            raise ValueError(f"File {path} is not a valid .encoder file.")
        # TODO: in the future, verify format version by checking
        # compatibility via _ENCODER_FILE_FORMAT_VERSION_COMPATIBILITY
        if state["encoder_class"] != cls.__name__:
            raise ValueError(
                f"File {path} was saved by {state['encoder_class']}. "
                f"Load it with {state['encoder_class']}.load_from_disk instead."
            )
        encoder = cls(
            feature_extractor=feature_extractor,
            similarity_func=similarity_func,
            power_norm_weight=state["power_norm_weight"],
            norm_order=state["norm_order"],
            epsilon=state["epsilon"],
            flatten=state["flatten"],
            raise_error_when_pca_incompatible=state[
                "raise_error_when_pca_incompatible"
            ],
        )
        if state["pca"] is not None:
            encoder.pca = state["pca"]
        encoder.clustering_model = state["clustering_model"]
        return encoder

    @_tupleize_first_arg
    def generate_encoding_map(self, image_paths: Iterable[str], /) -> ImageEncodingMap:
        """
        Build an :class:`~pyvisim.image_store.ImageEncodingMap` from image paths.

        The returned object is a lazy ``{image_path: encoded_vector}`` mapping:
        each image is read and encoded on first access and then buffered in
        memory.

        The result behaves like a regular dictionary: access the encoding for a path
        by simply:

        ```python
        image_path = "path/to/image.jpg"
        encoding = encoding_map[image_path]
        ```

        :param image_paths: List of image full paths.
        :return: An :class:`~pyvisim.image_store.ImageEncodingMap` mapping each
                image path to the descriptor vector of the corresponding image.
        """
        return ImageEncodingMap(self, image_paths)

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
            f"similarity_func={self.similarity_func.__name__}, \n"
            f"Number of cluster={n_clusters}, \n"
            f"Power Norm Weight={self.power_norm_weight}, \n"
            f"Norm Order={self.norm_order})"
        )
