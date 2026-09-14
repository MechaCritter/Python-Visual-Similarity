from typing import Any, ClassVar

import numpy as np

from ..base import SerializableImageEmbedder
from ..typing import (
    FloatNumpyArray,
    UInt8NumpyArray,
)


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
    :param normalize: Whether :meth:`embed` L2-normalizes the joined embeddings
        it returns.
    :param batch_size: Maximum number of images processed in a single batch.
        Set to ``-1`` to process all images as a single batch.
    :raises ValueError: If ``embedders`` is empty or holds anything but a
        SerializableImageEmbedder.
    """

    __format_version__: ClassVar[int] = 3
    __state_keys__: ClassVar[frozenset[str]] = (
        SerializableImageEmbedder.__state_keys__ | {"classic"}
    )

    def __init__(
        self,
        embedders: list[SerializableImageEmbedder],
        similarity_func: str = "cosine",
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ):
        self._check_valid_embedders(embedders)
        self.embedders = embedders
        super().__init__(
            similarity_func=similarity_func,
            normalize=normalize,
            batch_size=batch_size,
        )

    def _check_valid_embedders(
        self, embedders: list[SerializableImageEmbedder]
    ) -> None:
        """
        Checks that the pipeline holds at least one embedder and only
        instances of SerializableImageEmbedder.

        :param embedders: list of embedders to check.
        :raises ValueError: If ``embedders`` is empty or holds anything but a
            SerializableImageEmbedder.
        """
        if not embedders:
            raise ValueError("Pipeline needs at least one embedder, got none.")
        for embedder in embedders:
            if not isinstance(embedder, SerializableImageEmbedder):
                raise ValueError(
                    f"Pipeline only accepts instances of SerializableImageEmbedder, not {type(embedder)}"
                )

    def _state(self) -> dict[str, Any]:
        return {
            "classic": [embedder.to_dict() for embedder in self.embedders],
            "similarity_func": self._similarity_func_name,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> "Pipeline":
        cls._reject_unsupported_kwargs(kwargs)
        embedders = [
            SerializableImageEmbedder.from_dict(embedder_state)
            for embedder_state in state["classic"]
        ]
        return cls(
            embedders,
            similarity_func=state["similarity_func"],
            normalize=state["normalize"],
            batch_size=state["batch_size"],
        )

    def _embed(self, images: list[UInt8NumpyArray]) -> FloatNumpyArray:
        all_embeddings = []
        for metric in self.embedders:
            # Each embedder has to be flattened to be usable here. Embedders that
            # do not expose a ``flatten`` flag are assumed to already emit flat
            # ``(num_imgs, feature_dim)`` embeddings.
            has_flatten = hasattr(metric, "flatten")
            if has_flatten:
                original_flatten = metric.flatten  # type: ignore[attr-defined]
                metric.flatten = True  # type: ignore[attr-defined]
            try:
                # Each of size (num_imgs, feature_dim)
                all_embeddings.append(metric.embed(images))
            finally:
                if has_flatten:
                    metric.flatten = original_flatten  # type: ignore[attr-defined]
        # The embedders' vectors sit side by side, in the pipeline's order.
        return np.hstack(all_embeddings)

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
