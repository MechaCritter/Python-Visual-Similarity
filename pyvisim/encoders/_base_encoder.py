import abc
from collections.abc import Iterator, Iterable, MutableSequence
from enum import Enum
from functools import lru_cache, wraps
import warnings
from typing import Callable, Optional, Any

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

from .._config import setup_logging, PICKLE_MODEL_FILES_PATH
from ..features._features import FeatureExtractorBase
from .._base_classes import SimilarityMetric

setup_logging()


def check_desired_output(
    similarity_func: Callable[[np.ndarray, np.ndarray], Any],
    vecs1: np.ndarray,
    vecs2: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
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
        warnings.warn(f"Similarity function threw an error: {e}. Falling back to row-wise loop.")
        return _make_fallback_func(similarity_func)

    if not isinstance(out, np.ndarray):
        warnings.warn(f"Expected a NumPy array, got {type(out)}. Using fallback method.")
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
        warnings.warn(f"Output shape {out.shape} is not the expected (N, M). Expected output shape to be "
                                f"({vecs1.shape[0]}, {vecs2.shape[0]}). Using fallback.")
        return _make_fallback_func(similarity_func)

    return similarity_func

def _make_fallback_func(sim_func: Callable[[np.ndarray, np.ndarray], Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a new function that loops row-by-row if the original
    similarity function can't handle batch mode.
    """
    def fallback(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        N = vecs1.shape[0] # (N, D)
        M = vecs2.shape[0] # (M, D)
        out = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                out[i, j] = sim_func(vecs1[i:i+1], vecs2[j:j+1])
        return out
    try:
        return fallback
    except Exception as e:
        raise RuntimeError(f"Row-wise operation was not possible with the given similarity function: {e}"
                         "Your function is invalid.")

def _tupleize_first_arg(func: Callable) -> Callable:
    """
    # TODO: currently, the param 'image_paths' param is hardcoded. This should be more general
    # TODO: to be able to handle any variable name
    Pass this to the "encode" and "generate_encoding_map" methods to
    convert the input to a tuple so that it can be hashed by the lru_cache.
    """
    @wraps(func)
    def wrapper(self, image_paths: Any, /, *args, **kwargs):
        if isinstance(image_paths, (Iterator, MutableSequence)):
            image_paths = tuple(image_paths)
        return func(self, image_paths, *args, **kwargs)
    return wrapper

# TODO: add weights for SIFT and ROOTSIFT also
class _PretrainedModels(Enum):
    def load(self) -> object:
        """Loads the model from the file path"""
        with open(self.value, 'rb') as f:
            return joblib.load(f)


class KMeansWeights(_PretrainedModels):
    OXFORD102_K256_VGG16_PCA = f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_deep_features_vgg16_pca.pkl"
    OXFORD102_K256_VGG16 = f"{PICKLE_MODEL_FILES_PATH}/k_means_k256_deep_features_vgg16_no_pca.pkl"

class _KMeansPCA(_PretrainedModels):
    OXFORD102_PCA256_VGG16 = f"{PICKLE_MODEL_FILES_PATH}/pca_kmeans_k256_deep_features_vgg16_f2.pkl"

class GMMWeights(_PretrainedModels):
    OXFORD102_K256_VGG16_PCA = f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_deep_features_vgg16_pca.pkl"
    OXFORD102_K256_VGG16 = f"{PICKLE_MODEL_FILES_PATH}/gmm_k256_deep_features_vgg16_no_pca.pkl"

class _GMMPCA(_PretrainedModels):
    OXFORD102_PCA256_VGG16 = f"{PICKLE_MODEL_FILES_PATH}/pca_gmm_k256_deep_features_vgg16_f2.pkl"

_CLUSTERING_TO_PCA_MAPPING = {
    KMeansWeights.OXFORD102_K256_VGG16_PCA: _KMeansPCA.OXFORD102_PCA256_VGG16,
    GMMWeights.OXFORD102_K256_VGG16_PCA: _GMMPCA.OXFORD102_PCA256_VGG16
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
    :param weights: Pretrained model for clustering. If provided, the clustering model will be loaded from the file,
    and `clustering_model` and `pca` parameters will be ignored.
    :param clustering_model: Clustering model used for generating descriptors.
    :param power_norm_weight: Exponent for power normalization
    :param norm_order: Norm order for normalization (default: 2).
    :param epsilon: Small constant to avoid division by zero.
    :param flatten: Whether to flatten the computed descriptor vector (default: True).
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
    :param pca: PCA model for dimensionality reduction (optional).
    :param raise_error_when_pca_incompatible: When set to True, if the new clustering model has a different input size
                                        than the PCA model's output size, an Error will be raised """
    def __init__(
            self,
            feature_extractor: FeatureExtractorBase=None,
            weights: _PretrainedModels=None,
            clustering_model=None,
            similarity_func: Callable[[np.ndarray, np.ndarray], float]=None,
            power_norm_weight: float=1,
            norm_order: int = 2,
            epsilon: float = 1e-9,
            flatten: bool = True,
            pca: Optional[PCA] = None,
            raise_error_when_pca_incompatible: bool = True
    ):
        # Set important attributes via setters to trigger error handling
        self._feature_extractor = None
        self._clustering_model = None
        self._pca = None
        self._similarity_func = None

        self.similarity_func = similarity_func
        self.feature_extractor = feature_extractor

        if weights is not None:
            self.clustering_model = weights.load()
            if 'PCA' in weights.name:
                self.pca = _CLUSTERING_TO_PCA_MAPPING[weights].load()
        else:
            if clustering_model is not None:
                self.clustering_model = clustering_model
            if pca is not None:
                self.pca = pca

        self.power_norm_weight = power_norm_weight
        self.norm_order = norm_order
        self.epsilon = epsilon
        self.flatten = flatten
        self.raise_error_when_pca_incompatible = raise_error_when_pca_incompatible

    @property
    def feature_extractor(self) -> FeatureExtractorBase:
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, feature_extractor: FeatureExtractorBase):
        if not isinstance(feature_extractor, FeatureExtractorBase):
            raise TypeError(f"feature_extractor must be an instance of FeatureExtractorBase, not {type(feature_extractor)}")
        if self._pca is not None:
            if feature_extractor.output_dim != self._pca.n_features_in_:
                raise RuntimeError(f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                                   f"But PCA accepts input dim {self._pca.n_features_in_}")
        else:
            if self._clustering_model is not None:
                if feature_extractor.output_dim != self._clustering_model.n_features_in_:
                    raise RuntimeError(f"Feature Extractor outputs shape {feature_extractor.output_dim}, "
                                       f"But clustering model accepts input dim {self._clustering_model.n_features_in_}")
        self._feature_extractor = feature_extractor

    @property
    def similarity_func(self):
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, func: Callable[[np.ndarray, np.ndarray], float]):
        dummy1, dummy2 = np.random.rand(10, 10), np.random.rand(10, 10)
        self._similarity_func = check_desired_output(func, dummy1, dummy2)

    @property
    def clustering_model(self):
        return self._clustering_model

    @clustering_model.setter
    def clustering_model(self, clustering_model):
        if self._pca:
            if self._pca.n_components != self._clustering_model.n_features_in_:
                if self.raise_error_when_pca_incompatible:
                    raise RuntimeError(f"PCA is incompatible with the new clustering model. "
                                    f"PCA input size: {self._pca.n_components}, "
                                    f"New clustering model input size: {self._clustering_model.n_features_in_}. "
                                    f"If you want the PCA to be reset to None instead, set raise_error_when_pca_incompatible=False.")
                warnings.warn(f"PCA is incompatible with the new clustering model. "
                              f"PCA input size: {self._pca.n_components}, "
                              f"New clustering model input size: {self._clustering_model.n_features_in_}. "
                              "PCA will be reset to None to avoid errors."
                              "If you want to raise an Error instead when this happens, set raise_error_when_pca_incompatible=False.")
                self._pca = None
        self._clustering_model = clustering_model

    @property
    def pca(self):
        return self._pca

    @pca.setter
    def pca(self, pca: PCA):
        if not self._clustering_model:
            raise ValueError("PCA cannot be set without an existing clustering model.")

        if pca.n_features_in_ != self._feature_extractor.output_dim:
            raise ValueError("PCA input size has to match the feature extractor output size."
                             f"PCA model has input size {pca.n_features_in_}, "
                             f"while feature extractor has output size {self._feature_extractor.output_dim}")

        if pca.n_components != self._clustering_model.n_features_in_:
            raise ValueError("PCA input size has to match the clustering model input size."
                             f"PCA model has input size {pca.n_components}, "
                             f"while clustering model has input size {self._clustering_model.n_features_in_}")

        self._pca = pca

    def learn(self, images: Iterable[np.ndarray], /, *, n_clusters: int, dim_reduction_factor: int=None, **kwargs) -> None:
        """
        Learns the visual vocabulary from a collection of images.

        :param images: An iterable of images. First argument of the iterable is expected to be the image itself.
        :param n_clusters: Number of clusters to use for the clustering model
        :param dim_reduction_factor: If a value is provided, PCA will be used to reduce the dimensionality of the feature space
        :param kwargs: Additional arguments for the clustering model
        """
        features = np.vstack([self.feature_extractor(image) for image, *_ in images])
        print("[INFO] Learning the visual vocabulary with the following parameters:")
        print("   - Number of clusters:", n_clusters)
        print("   - Feature Extractor used:", self.feature_extractor.__class__.__name__)
        print("   - Dimension of the feature space:", feat_dim := features.shape[1])
        if dim_reduction_factor:
            print("   - New dimension after PCA reduction:", new_dim := feat_dim // dim_reduction_factor)
            self._pca = PCA(n_components=new_dim)
            self._pca.fit(features)
            features = self._pca.transform(features)
        clustering_model = KMeans(n_clusters=n_clusters, **kwargs)
        clustering_model.fit(features)
        self.clustering_model = clustering_model

    @_tupleize_first_arg
    @lru_cache(maxsize=4)
    def generate_encoding_map(self, image_paths: Iterable[str], /) -> dict[str, np.ndarray]:
        """
        Converts a collection of image file paths into a dictionary of
        ``{image_path: encoded_vector}``.

        This method automatically reads each image, applies the internal
        encoding pipeline, and stores the resulting descriptor vector.

        :param image_paths: List of image full paths
        :return: a dictionary where keys are image paths and values are descriptor vectors of the
                corresponding images
        """
        images = (cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths)
        return dict(zip(image_paths, self.encode(images)))

    @abc.abstractmethod
    def encode(self, images: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Encodes one or more images into a batch of vector representations.

        :param images: iterable. Consider using an iterator if you have a lot of images.
        :return: vector representations of the given images
        """
        raise NotImplementedError

    def similarity_score(self, images1: Iterable[np.ndarray] | np.ndarray, images2: Iterable[np.ndarray] | np.ndarray) -> float:
        """
        Computes vector encodings for two images and calculates the similarity score between them.

        :param images1: First (batch of) image(s)
        :param images2: Second (batch of) image(s)
        :return: Similarity score. If image iterables are provided, a similarity matrix between two image batches is returned.
        """
        vector1 = self.encode(images1)
        vector2 = self.encode(images2)
        result = self.similarity_func(vector1, vector2)
        return np.float_(result)

    def __repr__(self) -> str:
        n_clusters = None
        if self._clustering_model:
            if hasattr(self._clustering_model, 'n_clusters'):
                n_clusters = self._clustering_model.n_clusters
            elif hasattr(self._clustering_model, 'n_components'):
                n_clusters = self._clustering_model.n_components
        return self.__class__.__name__ + f"(feature_extractor={self.feature_extractor.__class__.__name__}, \n" \
               f"similarity_func={self.similarity_func.__name__}, \n" \
               f"Number of cluster={n_clusters}, \n" \
               f"Power Norm Weight={self.power_norm_weight}, \n" \
               f"Norm Order={self.norm_order})"