from collections.abc import Iterator
import logging
from itertools import tee
from typing import Iterable, Callable

import numpy as np
import torch
import cv2

from ..encoders._base_encoder import ImageEncoderBase, SimilarityMetric
from .._utils import cosine_similarity
from ._base_encoder import check_desired_output


class Pipeline(SimilarityMetric):
    """
    A pipeline for computing feature vectors using a set of
    descriptor-based encoders (e.g., VLAD, Fisher, etc.).

    Currently, all vectors computed using the Encoders listed
    will always be flattened, because different Encoders also
    have different output sizes.

    :param encoders: A list of ImageEncoderBase instances.
    :param similarity_func: A function that takes two batches of vectors and returns a similarity score
    matrix with size (batch_1_size, batch_2_size).
    """
    _logger = logging.getLogger("Pipeline")
    def __init__(
            self,
            encoders: list[ImageEncoderBase],
            similarity_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
    ):
        self._check_valid_encoders(encoders)
        self.encoders = encoders
        self._similarity_func = similarity_func

    def _check_valid_encoders(self, encoders: list[ImageEncoderBase]) -> None:
        """
        Checks if all encoders in the pipeline are instances of ImageEncoderBase.
        :param encoders: list of encoders to check.
        """
        for encoder in encoders:
            if not isinstance(encoder, ImageEncoderBase):
                raise ValueError(f"Pipeline only accepts instances of ImageEncoderBase, not {type(encoder)}")

    def encode(self, images: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Encode an image using all encoders in the pipeline.

        :param images: Images to process.
        :return: encoded images using the combined encoders.
        """
        all_encodings = []
        if isinstance(images, torch.Tensor):
            raise RuntimeError("Torch images are not supported yet.")
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images] # Handle single image case
        images_gen = tee(images, len(self.encoders))
        for metric, images in zip(self.encoders, images_gen):
            a = metric.flatten # each encoder has to be flattened to be usable here. Saving the original state temporarily
            metric.flatten = True
            encodings = metric.encode(images) # Each of size (num_imgs, feature_dim)
            all_encodings.append(encodings)
            metric.flatten = a
        return np.hstack(all_encodings)

    def generate_encoding_map(self, image_paths: Iterable[str]) -> dict[str, np.ndarray]:
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

    @property
    def similarity_func(self):
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, func: Callable[[np.ndarray, np.ndarray], float]):
        dummy1, dummy2 = np.random.rand(10, 10), np.random.rand(10, 10)
        self._similarity_func = check_desired_output(func, dummy1, dummy2)

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
        return np.float32(result)

    # def fit(self, images: Iterable[np.ndarray], reduce_dimension: bool = False, reduce_factor: int=2) -> None:
    #     """
    #     Trains any clustering model_files used by the encoders in this pipeline, if they have a fit method.
    #
    #     :param images: Iterable of images (NumPy arrays) used for fitting the pipeline's encoders.
    #     :param reduce_dimension: Whether to apply dimension reduction (e.g., PCA) if supported.
    #     :param reduce_factor: Factor to reduce the dimension by.
    #     """
    #     for metric in self.encoder:
    #         if hasattr(metric, 'fit') and callable(metric.fit):
    #             self._logger.info(f"Fitting {metric.__class__.__name__} with reduce_dimension={reduce_dimension}...")
    #             metric.fit(images, reduce_dimension=reduce_dimension, reduce_factor=reduce_factor)
    #         else:
    #             self._logger.warning(f"{metric.__class__.__name__} has no 'fit' method. Skipping...")

    def __repr__(self) -> str:
        """
        Returns a string representation of this Pipeline, including the names
        of the encoders and the similarity function used.
        """
        encoders_str = "\n".join([str(encoder) for encoder in self.encoders])
        return (f"Pipeline(\n"
                f"encoders=[{encoders_str}],\n"
                f"similarity_func={self._similarity_func.__name__ if hasattr(self._similarity_func, '__name__') else str(self._similarity_func)})")