"""
Includes exceptions for the package.
"""


class InvalidImageError(Exception):
    """
    Raised when an image is not provided.
    """

    def __init__(self, message: str = "Input is not a valid image."):
        super().__init__(message)


class ChecksumMismatchError(Exception):
    """
    Raised when a downloaded or cached file fails its SHA-256 integrity check.
    """

    def __init__(self, message: str = "File failed its SHA-256 integrity check."):
        super().__init__(message)
