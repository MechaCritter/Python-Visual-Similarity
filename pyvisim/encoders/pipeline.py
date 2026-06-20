import logging
from typing import Any, cast

import numpy as np

from .._base_classes import SimilarityMetric
from .._utils import get_similarity_func
from ..typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
)
from ._base_encoder import ImageEncoderBase
from .utils import iter_images

#: On-disk format version of the serialised pipeline state.
_PIPELINE_FORMAT_VERSION = 1


class Pipeline(SimilarityMetric):
    """
    A pipeline for computing feature vectors using a set of
    descriptor-based encoders (e.g., VLAD, Fisher, etc.).

    Currently, all vectors computed using the Encoders listed
    will always be flattened, because different Encoders also
    have different output sizes.

    :param encoders: A list of ImageEncoderBase instances.
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    """

    _logger = logging.getLogger("Pipeline")

    def __init__(
        self,
        encoders: list[ImageEncoderBase],
        similarity_func: str = "cosine",
    ):
        self._check_valid_encoders(encoders)
        self.encoders = encoders
        self.similarity_func = similarity_func

    def _check_valid_encoders(self, encoders: list[ImageEncoderBase]) -> None:
        """
        Checks if all encoders in the pipeline are instances of ImageEncoderBase.
        :param encoders: list of encoders to check.
        """
        for encoder in encoders:
            if not isinstance(encoder, ImageEncoderBase):
                raise ValueError(
                    f"Pipeline only accepts instances of ImageEncoderBase, not {type(encoder)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the pipeline into a JSON-safe state dictionary.

        Each member encoder is serialised with its own
        :meth:`~pyvisim.encoders.ImageEncoderBase.to_dict`, so every encoder
        must be fitted.

        :return: A JSON-safe pipeline description suitable for
            :meth:`from_dict`.
        :raises NotFittedError: If any member encoder is not fitted.
        """
        return {
            "format_version": _PIPELINE_FORMAT_VERSION,
            "encoder_class": type(self).__name__,
            "encoders": [encoder.to_dict() for encoder in self.encoders],
            "similarity_func": self._similarity_func_name,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "Pipeline":
        """
        Rebuilds a pipeline from a dictionary produced by :meth:`to_dict`.

        :param state: A JSON-safe pipeline description from :meth:`to_dict`.
        :return: A ready-to-use :class:`Pipeline` instance.
        """
        # Imported lazily to avoid an import cycle with the encoder registry.
        from ._reconstruct import encoder_from_dict

        encoders = [
            cast(ImageEncoderBase, encoder_from_dict(encoder_state))
            for encoder_state in state["encoders"]
        ]
        return cls(encoders, similarity_func=state["similarity_func"])

    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Encode an image using all encoders in the pipeline.

        The input is normalized once into canonical ``uint8`` ``(H, W, C)``
        images (handling torch tensors and batches), then shared across every
        encoder in the pipeline.

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
        :return: encoded images using the combined encoders.
        """
        image_list = list(iter_images(images, dims=dims, value_range=value_range))
        all_encodings = []
        for metric in self.encoders:
            a = metric.flatten  # each encoder has to be flattened to be usable here. Saving the original state temporarily
            metric.flatten = True
            encodings = metric.encode(
                image_list
            )  # Each of size (num_imgs, feature_dim)
            all_encodings.append(encodings)
            metric.flatten = a
        return np.hstack(all_encodings)

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
        return (
            f"Pipeline(\n"
            f"encoders=[{encoders_str}],\n"
            f"similarity_func={self._similarity_func_name})"
        )
