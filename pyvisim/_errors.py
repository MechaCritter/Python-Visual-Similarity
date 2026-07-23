"""
Includes exceptions for the package.
"""


class NotFittedError(ValueError, AttributeError):
    """Raised when a model is used before it has been fitted."""


class InvalidImageError(Exception):
    """
    Raised when an image is not provided.
    """

    def __init__(self, message: str = "Input is not a valid image."):
        super().__init__(message)
