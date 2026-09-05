"""Structural type describing the embedder interface used across pyvisim."""

from typing import Protocol, runtime_checkable

from .numeric import FloatNumpyArray, ImageInput


@runtime_checkable
class Embedder(Protocol):
    """Protocol for objects that embed images into fixed-size vectors."""

    @property
    def batch_size(self) -> int:
        """
        Maximum number of images the embedder processes in a single batch.

        ``-1`` means the whole input is processed as one batch.
        """
        ...

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = ...,
        value_range: tuple[float, float] = ...,
    ) -> FloatNumpyArray:
        """
        Embed one or more images into a batch of vector representations.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images.
        :param dims: Axis-label string describing the input axes (see
            :mod:`pyvisim.typing`).
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: A ``(N, D)`` array holding one ``D``-dimensional embedding per
            input image.
        """
        ...
