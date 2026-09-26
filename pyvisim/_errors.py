"""
Includes exceptions for the package.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MissingModuleFromExtraMessage:
    """
    Message telling the user to install the extra that provides a missing module.

    :param module: Import name of the missing module, e.g. ``"torch"``.
    :param extra: Name of the pip/uv extra that installs it, e.g. ``"nn"``.
    """

    module: str
    extra: str

    def __str__(self) -> str:
        """
        Render the install instructions.

        :returns: The message naming the missing module and the install commands.
        """
        return (
            f"To use this feature, you need to install the optional dependency '{self.module}'. "
            f"Install it with: 'uv pip install \"pyvisim[{self.extra}]\"' or 'pip install \"pyvisim[{self.extra}]\"'"
        )


class NotFittedError(ValueError, AttributeError):
    """
    Raised when a model is used before it has been fitted.

    Inherits from both :class:`ValueError` and :class:`AttributeError` to
    keep the hierarchy of :class:`sklearn.exceptions.NotFittedError`, which
    this exception replaces, so existing ``except`` clauses keep working.
    """


class InvalidParameterError(ValueError, TypeError):
    """
    Raised when an argument has the wrong type or lies outside its valid range.

    Inherits from both :class:`ValueError` and :class:`TypeError`, so an
    ``except`` clause written for either one keeps catching it.
    """


class InvalidImageError(Exception):
    """
    Raised when an image is not provided.
    """

    def __init__(self, message: str = "Input is not a valid image."):
        super().__init__(message)
