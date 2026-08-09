"""
Includes exceptions for the package.
"""


class NotFittedError(ValueError, AttributeError):
    """
    Raised when a model is used before it has been fitted.

    Inherits from both :class:`ValueError` and :class:`AttributeError` to
    keep the hierarchy of :class:`sklearn.exceptions.NotFittedError`, which
    this exception replaces, so existing ``except`` clauses keep working.
    """


class InvalidImageError(Exception):
    """
    Raised when an image is not provided.
    """

    def __init__(self, message: str = "Input is not a valid image."):
        super().__init__(message)
