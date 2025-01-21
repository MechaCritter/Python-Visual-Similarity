"""
Includes exceptions for the package.
"""

class InvalidImageError(Exception):
    """
    Raised when an image is not provided.
    """
    def __init__(self, message: str = "Input is not a valid image."):
        super().__init__(message)