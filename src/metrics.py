from dataclasses import dataclass, field
import logging

import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.feature import fisher_vector
import numpy as np
import cv2
from typing import List
from piq import ssim, multi_scale_ssim as ms_ssim

from src.utils import sift, root_sift, load_model
from src.config import setup_logging

setup_logging()

__all__ = ["VLAD", "FisherVector", "SSIM", "MS_SSIM"]

_logger_vv = logging.getLogger("VLAD_Vector")
_logger_fv = logging.getLogger("Fisher_Vector")


@dataclass
class ClusteringBasedMetric:
    """
    TODO: remind user to specify 'sift' or 'root_sift' as feature when using VLAD or Fisher Vector based on the clustering model loaded.
    Base class for clustering-based metrics (metrics that use k-means or GMM). Used for VLAD and Fischer Vector calculations.

    - **Note**: All the attributes are read-only. They are calculated internally and should not be modified.

    The class has following modifiable attributes:
    ----------------------------------------------
        - np.ndarray image: The image for which to calculate the metrics.
        - int norm_order: The order of the norm to use for normalization. Default is 2 (l2 norm will be applied in this case).
        - float power_norm_weight: The weight to apply to the power normalization. Default is 0.5.
        - float epsilon: A small value to add to the denominator to avoid division by zero.
        - bool flatten: Whether to flatten the resulting vector (the vector becomes 1D). Default is False.
        - bool verbose: Whether to print the keypoints data, descriptors, and other information. Default is False.
        - str feature: The feature to use for the image. Default is "sift". Accepted values: "sift", "root_sift".

    Please do not modify attributes stated below:
    --------------------------------------------

    :ivar keypoints: List of cv2.KeyPoint objects.
    :vartype keypoints: List[cv2.KeyPoint]
    :ivar descriptors: Descriptors of the keypoints (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptors: np.ndarray
    :ivar descriptor_centroids: centroids of the descriptors (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptor_centroids: np.ndarray
    :ivar descriptor_labels: Labels of the descriptors (which cluster/centroid they belong to).
    :vartype descriptor_labels: np.ndarray
    """
    image: np.ndarray = field(repr=False)
    norm_order: int = 2
    power_norm_weight: float = 0.5
    epsilon: float = 1e-9
    flatten: bool = True
    verbose: bool = False
    feature: str = "sift"

    keypoints: List[cv2.KeyPoint] = field(init=False, repr=False)
    descriptors: np.ndarray = field(init=False)
    descriptor_centroids: np.ndarray = field(init=False)
    descriptor_labels: np.ndarray = field(init=False)
    keypoints_2d_coords: np.ndarray = field(init=False)
    keypoints_labels: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray):
            raise ValueError(f"Image must be a numpy array, not {type(self.image)}")
        if self.power_norm_weight < 0 or self.power_norm_weight > 1:
            raise ValueError("Power norm weight must be between 0 and 1.")
        self.get_sift()
        self.get_keypoint_coords()

    def get_sift(self):
        """
        Get the SIFT features for the image. This includes the keypoints (with all their attributes like angle, label, ...) and descriptors (128-dimensional vectors).
        """
        if self.feature == "sift":
            self.keypoints, self.descriptors = sift(self.image)
        elif self.feature == "root_sift":
            self.keypoints, self.descriptors = root_sift(self.image)
        else:
            raise ValueError("Feature must be 'sift' or 'root_sift', not {self.feature}")

    def get_keypoint_coords(self):
        """
        Get the 2D coordinates of the keypoints.
        """
        self.keypoints_2d_coords = np.array([kp.pt for kp in self.keypoints])

@dataclass
class VLAD(ClusteringBasedMetric):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) for the given image.

    To retrieve to VLAD vector, simply call the `vector` attribute of the object.

    :param k_means: The pre-trained k-means model to use for clustering the descriptors.
    :type k_means: KMeans

    :ivar _vector: The VLAD vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    """
    k_means: KMeans = None
    _vector: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.centroids_and_labels()
        self.compute_vlad_vector()

        if self.verbose:
            print(
                "====================================\n"
                "Vector Type: ", self.__class__.__name__, "\n"
                "Keypoints data:\n"
                "Number of keypoints: \n", len(self.keypoints), "\n"
                "Keypoint angles: \n", [kp.angle for kp in self.keypoints], "\n"
                "Keypoint sizes: \n", [kp.size for kp in self.keypoints], "\n"
                "Keypoint responses: \n", [kp.response for kp in self.keypoints], "\n"
                "Keypoint octaves: \n", [kp.octave for kp in self.keypoints], "\n"
                "Keypoint class IDs: \n", [kp.class_id for kp in self.keypoints], "\n"
                "====================================\n"
                "Descriptor centroids: \n", self.descriptor_centroids, "\n"
                "====================================\n"
                "Descriptor vectors: \n", self.descriptors, "\n"
                "Number of descriptor vectors: \n", len(self.descriptors), "\n"
                "Length of one descriptor vector: \n", len(self.descriptors[0]), "\n"
                "====================================\n"
            )

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    def centroids_and_labels(self):
        self.descriptor_labels = self.k_means.predict(self.descriptors.astype(np.float32))
        self.descriptor_centroids = self.k_means.cluster_centers_

    def compute_vlad_vector(self) -> None:
        """
        Compute the VLAD descriptor for the image. Each VLAD vector has a fixed size of
        128, as calculated using OpenCV's SIFT_create() function. After calculaing the VLAD
        vector, normalization is applied to the vector.
        """
        vlad = np.zeros((len(self.k_means.cluster_centers_), 128))

        for i in range(len(self.descriptors)):
            label = self.descriptor_labels[i]
            vlad[label] += self.descriptors[i] - self.descriptor_centroids[label]
        _logger_vv.debug("VLAD vector before normalization: %s", vlad)

        # Power normalization
        vlad = np.sign(vlad) * (np.abs(vlad) ** self.power_norm_weight)
        _logger_vv.debug("VLAD vector after power normalization: %s", vlad)

        # L2 normalization (if norm_order = 2)
        norm = np.linalg.norm(vlad, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
        _logger_vv.debug("Norm vector of VLAD vector: %s", norm)

        vlad = vlad / norm
        _logger_vv.debug("VLAD vector after L2 normalization: %s", vlad)

        if self.flatten:
            vlad = vlad.flatten()
            _logger_vv.debug("Flattened VLAD vector: %s", vlad)

        self._vector = vlad
        _logger_vv.info("Resulting VLAD vector: %s. Shape of vector: %s", self._vector, self._vector.shape)

@dataclass
class FisherVector(ClusteringBasedMetric):
    """
    Calculate the Fischer Vector for the given image. For D-dimensional input descriptors or vectors, and a K-mode GMM, 
    the Fisher vector dimensionality will be 2KD + K. Thus, its dimensionality is invariant to the number of descriptors/vectors.
    
    To retrieve the Fischer Vector, simply call the `vector` attribute of the object.

    **Attributes**:

    :ivar _vector: The Fischer Vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    :ivar gmm: The Gaussian Mixture Model used to calculate the Fischer Vector.
    :vartype gmm: GaussianMixture
    """
    gmm: GaussianMixture = None
    _vector: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.centroids_and_labels()
        self.compute_fisher_vector()

        if self.verbose:
            print(
                "====================================\n"
                "Vector Type: ", self.__class__.__name__, "\n"
                "Keypoints data:\n"
                "Number of keypoints: \n", len(self.keypoints), "\n"
                "Keypoint angles: \n", [kp.angle for kp in self.keypoints], "\n"
                "Keypoint sizes: \n", [kp.size for kp in self.keypoints], "\n"
                "Keypoint responses: \n", [kp.response for kp in self.keypoints], "\n"
                "Keypoint octaves: \n", [kp.octave for kp in self.keypoints], "\n"
                "Keypoint class IDs: \n", [kp.class_id for kp in self.keypoints], "\n"
                "====================================\n"
                "Descriptor centroids: \n", self.descriptor_centroids, "\n"
                "====================================\n"
                "Descriptor vectors: \n", self.descriptors, "\n"
                "Length of one descriptor vector: \n", len(self.descriptors[0]), "\n"
                "====================================\n"
            )

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    def centroids_and_labels(self):
        self.descriptor_labels = self.gmm.predict(self.descriptors.astype(np.float32))
        self.descriptor_centroids = self.gmm.means_

    def compute_fisher_vector(self) -> None:
        """
        Compute the Fischer Vector for the image. The Fisher Vector is calculated using the
        Fisher Vector algorithm from the scikit-image library.
        """
        # Extract the Fisher Vector
        self._vector = np.array([fisher_vector(self.descriptors, self.gmm, alpha=self.power_norm_weight)])

        _logger_fv.debug("Fisher vector before normalization: %s", self._vector)

        # Power normalization
        self._vector = np.sign(self._vector) * (np.abs(self._vector) ** self.power_norm_weight)
        _logger_fv.debug("Fisher vector after power normalization: %s", self._vector)

        # L2 normalization
        norm = np.linalg.norm(self._vector, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
        _logger_fv.debug("Norm vector of Fisher vector: %s", norm)
        self._vector = self._vector / norm

        _logger_fv.debug("Fisher vector after L2 normalization: %s", self._vector)

        if self.flatten:
            self._vector = self._vector.flatten()
            _logger_fv.debug("Flattened Fischer vector: %s", self._vector)

        _logger_fv.info("Resulting Fischer Vector: %s. Shape of vector: %s", self._vector, self._vector.shape)

@dataclass
class StructuralSimilarity:
    """
    Base class for SSIM and MS-SSIM metrics. Used for comparing two images. Pass images with shape (C, H, W) as input.

    :image_1: The first image to compare. Shape: (C, H, W).
    :image_2: The second image to compare. Shape: (C, H, W).
    :data_range: 1.0 for float images, 255 for uint8 images.
    """
    image_1: torch.Tensor = field(repr=False)
    image_2: torch.Tensor = field(repr=False)
    data_range: float = 1.0

@dataclass
class SSIM(StructuralSimilarity):
    """
    Calculate the Structural Similarity Index (SSIM) between two images. Pass normal RGB images as input.
    To access the SSIM value, call the `value` attribute of the object.
    """
    _ssim: torch.Tensor= field(init=False)
    def __post_init__(self):
        self._ssim = ssim(self.image_1.unsqueeze(0), self.image_2.unsqueeze(0), data_range=self.data_range)

    @property
    def value(self):
        return self._ssim

@dataclass
class MS_SSIM(StructuralSimilarity):
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two images. Pass normal RGB images as input.
    To access the MS-SSIM value, call the `value` attribute of the object.
    """
    _ms_ssim: torch.Tensor= field(init=False)
    def __post_init__(self):
        self._ms_ssim = ms_ssim(self.image_1.unsqueeze(0), self.image_2.unsqueeze(0), data_range=self.data_range)

    @property
    def value(self):
        return self._ms_ssim


if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity
    from src.datasets import *
    data = ExcavatorDataset(plot=True)
    image1 = data[1].image_array
    image2 = data[2].image_array
    k_means = load_model("models/pickle_model_files/k_means_model_k64_root_sift.pkl")
    gmm = load_model("models/pickle_model_files/gmm_model_k64_root_sift.pkl")
    pca_model = load_model("models/pickle_model_files/pca_model.pkl")

    vlad1 = VLAD(image=image1,
                 k_means=k_means,
                 feature="root_sift",
                 flatten=True).vector
    vlad2 = VLAD(image=image2,
                 k_means=k_means,
                feature="root_sift",
                 flatten=True).vector
    print("VLAD similarity between first two images:", cosine_similarity(vlad1.reshape(1, -1), vlad2.reshape(1, -1)))
    fisher1 = FisherVector(image=image1,
                           gmm=gmm,
                           feature="root_sift",
                           flatten=True).vector
    fisher2 = FisherVector(image=image2,
                           gmm=gmm,
                            feature="root_sift",
                           flatten=True).vector
    print("Fisher Vector similarity between first two images:", cosine_similarity(fisher1.reshape(1, -1), fisher2.reshape(1, -1)))



