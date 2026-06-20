"""
Structural type describing the encoder interface used across pyvisim.
"""

from typing import Protocol, runtime_checkable

from .numeric import FloatNumpyArray, ImageInput


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for objects that encode images into fixed-size vectors.

    An encoder turns one or more images into a batch of numeric feature
    vectors. Only :meth:`encode` is required; this keeps the protocol decoupled
    from any specific encoder implementation.
    """

    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = ...,
        value_range: tuple[float, float] = ...,
    ) -> FloatNumpyArray:
        """
        Encode one or more images into a batch of vector representations.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images.
        :param dims: Axis-label string describing the input axes (see
            :mod:`pyvisim.typing`).
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: A ``(N, D)`` array holding one ``D``-dimensional encoding per
            input image.
        """
        ...
