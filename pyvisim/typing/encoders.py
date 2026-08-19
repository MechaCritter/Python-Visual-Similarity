"""
Structural type describing the embedder interface used across pyvisim.
"""

from typing import Protocol, runtime_checkable

from .numeric import FloatNumpyArray, ImageInput


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for objects that embed images into fixed-size vectors.

    An embedder turns one or more images into a batch of numeric feature
    vectors. Only :meth:`embed` is required; this keeps the protocol decoupled
    from any specific implementation.
    """

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
