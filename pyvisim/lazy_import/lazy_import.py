"""Utilities for deferring the import errors of optional dependencies."""

from __future__ import annotations

from types import TracebackType


class OptionalImport:
    """Defer the :class:`ImportError` of an optional dependency.

    Use as a context manager around the imports of an optional dependency. If
    the dependency is installed, the imports run normally. If it is missing,
    the error is swallowed and stored, then re-raised with ``err_msg`` the
    first time :meth:`check` is called.

    :param err_msg: Message of the :class:`ImportError` raised by :meth:`check`
        when the dependency is missing.

    :Example:

    .. code-block:: python

        with OptionalImport(
            err_msg="To use this feature, you need to install the optional dependency 'transformers'."
        ) as _import:
            from transformers import AutoModel

        class Embedder:
            def __init__(self) -> None:
                _import.check()  # raises here if transformers is missing
    """

    def __init__(self, *, err_msg: str) -> None:
        self._err_msg = err_msg
        self._error: ImportError | None = None

    def __enter__(self) -> OptionalImport:
        """Enter the context manager.

        :returns: This instance, so it can be bound with ``as``.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Capture an :class:`ImportError` raised inside the ``with`` block.

        :param exc_type: Type of the raised exception, if any.
        :param exc_value: The raised exception instance, if any.
        :param traceback: The associated traceback, if any.
        :returns: ``True`` to suppress a captured :class:`ImportError`;
            ``False`` so any unrelated exception propagates normally.
        """
        if isinstance(exc_value, ImportError):
            self._error = exc_value
            return True
        return False

    @property
    def is_available(self) -> bool:
        """Whether the optional dependency imported successfully.

        :returns: ``True`` if the dependency is installed, ``False`` otherwise.
        """
        return self._error is None

    def check(self) -> None:
        """Re-raise the deferred import error with ``err_msg``.

        :raises ImportError: If the optional dependency failed to import.
        """
        if self._error is not None:
            raise ImportError(self._err_msg) from self._error
