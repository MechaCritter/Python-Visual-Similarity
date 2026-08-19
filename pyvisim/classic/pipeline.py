import logging
from typing import Any, ClassVar, cast

import numpy as np

from ..typing import (
    FloatNumpyArray,
    ImageInput,
)
from ..utils.image_utils import iter_images
from ._base_embedder import SerializableImageEmbedder

#: On-disk format version of the serialised pipeline state.
_PIPELINE_FORMAT_VERSION = 1


class Pipeline(SerializableImageEmbedder):
    """
    A pipeline for computing feature vectors using a set of
    embedders.

    Currently, all vectors computed using the Embedders listed
    will always be flattened, because different Embedders also
    have different output sizes.

    :param embedders: A list of SerializableImageEmbedder instances.
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    """

    _logger = logging.getLogger("Pipeline")

    #: Keys a serialised state must contain to be a valid pipeline file.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"embedder_class", "classic", "similarity_func"}
    )

    def __init__(
        self,
        embedders: list[SerializableImageEmbedder],
        similarity_func: str = "cosine",
    ):
        self._check_valid_embedders(embedders)
        self.embedders = embedders
        super().__init__(similarity_func=similarity_func)

    def _check_valid_embedders(
        self, embedders: list[SerializableImageEmbedder]
    ) -> None:
        """
        Checks if all embedders in the pipeline are instances of SerializableImageEmbedder.
        :param embedders: list of embedders to check.
        """
        for embedder in embedders:
            if not isinstance(embedder, SerializableImageEmbedder):
                raise ValueError(
                    f"Pipeline only accepts instances of SerializableImageEmbedder, not {type(embedder)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the pipeline into a JSON-safe state dictionary.

        Each member embedder is serialised with its own
        :meth:`~pyvisim.classic.SerializableImageEmbedder.to_dict`, so every embedder
        must be fitted.

        :return: A JSON-safe pipeline description suitable for
            :meth:`from_dict`.
        :raises NotFittedError: If any member embedder is not fitted.
        """
        return {
            "format_version": _PIPELINE_FORMAT_VERSION,
            "embedder_class": type(self).__name__,
            "classic": [embedder.to_dict() for embedder in self.embedders],
            "similarity_func": self._similarity_func_name,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "Pipeline":
        """
        Rebuilds a pipeline from a dictionary produced by :meth:`to_dict`.

        :param state: A JSON-safe pipeline description from :meth:`to_dict`.
        :return: A ready-to-use :class:`Pipeline` instance.
        """
        # Imported lazily to avoid an import cycle with the embedder registry.
        from ._reconstruct import embedder_from_dict

        embedders = [
            cast(SerializableImageEmbedder, embedder_from_dict(embedder_state))
            for embedder_state in state["classic"]
        ]
        return cls(embedders, similarity_func=state["similarity_func"])

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Embed an image using all embedders in the pipeline.

        The input is normalized once into canonical ``uint8`` ``(H, W, C)``
        images (handling torch tensors and batches), then shared across every
        embedder in the pipeline.

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
        :return: embedded images using the combined embedders.
        """
        image_list = list(iter_images(images, dims=dims, value_range=value_range))
        all_embeddings = []
        for metric in self.embedders:
            # Each embedder has to be flattened to be usable here. Embedders that
            # do not expose a ``flatten`` flag are assumed to already emit flat
            # ``(num_imgs, feature_dim)`` embeddings.
            has_flatten = hasattr(metric, "flatten")
            if has_flatten:
                original_flatten = metric.flatten  # type: ignore[attr-defined]
                metric.flatten = True  # type: ignore[attr-defined]
            embeddings = metric.embed(
                image_list
            )  # Each of size (num_imgs, feature_dim)
            all_embeddings.append(embeddings)
            if has_flatten:
                metric.flatten = original_flatten  # type: ignore[attr-defined]
        return np.hstack(all_embeddings)

    # def fit(self, images: Iterable[np.ndarray], reduce_dimension: bool = False, reduce_factor: int=2) -> None:
    #     """
    #     Trains any clustering model_files used by the embedders in this pipeline, if they have a fit method.
    #
    #     :param images: Iterable of images (NumPy arrays) used for fitting the pipeline's embedders.
    #     :param reduce_dimension: Whether to apply dimension reduction (e.g., PCA) if supported.
    #     :param reduce_factor: Factor to reduce the dimension by.
    #     """
    #     for metric in self.embedder:
    #         if hasattr(metric, 'fit') and callable(metric.fit):
    #             self._logger.info(f"Fitting {metric.__class__.__name__} with reduce_dimension={reduce_dimension}...")
    #             metric.fit(images, reduce_dimension=reduce_dimension, reduce_factor=reduce_factor)
    #         else:
    #             self._logger.warning(f"{metric.__class__.__name__} has no 'fit' method. Skipping...")

    def __repr__(self) -> str:
        """
        Returns a string representation of this Pipeline, including the names
        of the embedders and the similarity function used.
        """
        embedders_str = "\n".join([str(embedder) for embedder in self.embedders])
        return (
            f"Pipeline(\n"
            f"embedders=[{embedders_str}],\n"
            f"similarity_func={self._similarity_func_name})"
        )
