"""Shared save/load contract for the objects pyvisim writes to a safetensors file."""

from __future__ import annotations

import abc
import os
import pathlib
from collections.abc import Mapping
from typing import Any, ClassVar, TypeVar

from .serialization import load_state, save_state

_SerializableT = TypeVar("_SerializableT", bound="SerializerMixin")


class SerializerMixin(abc.ABC):
    """
    Mixin giving a self-describing object a safetensors file format.

    A subclass declares the kind of file it reads and writes through
    :attr:`__metadata_key__`, :attr:`__format_version__` and
    :attr:`__state_keys__`, and describes itself through :meth:`_state` /
    :meth:`from_dict`. In exchange it gets
    :meth:`save_to_disk` and :meth:`load_from_disk`, which are always
    `safetensors <https://github.com/huggingface/safetensors>`_ files: every
    NumPy array of the state is written as a binary tensor, the rest as a
    single JSON blob in the file's metadata.

    The steps in between are overridable hooks:
    :meth:`_resolve_save_path` and :meth:`_write_state` on the way out,
    :meth:`_read_state` and :meth:`_validate_state` on the way in.
    """

    #: Metadata key under which the state's JSON skeleton is stored in the file.
    __metadata_key__: ClassVar[str]
    #: On-disk format version, written into every state this class serializes.
    __format_version__: ClassVar[int]
    #: Keys a serialized state must contain to be a valid file of this kind,
    #: besides the format version and the class name :meth:`to_dict` adds.
    __state_keys__: ClassVar[frozenset[str]]
    #: Whether a file written under one format version can be read under
    #: another, keyed by ``(written version, reading version)``.
    __compatibility_mapping__: ClassVar[Mapping[tuple[int, int], bool]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Rejects a subclass that leaves a required class attribute undeclared."""
        super().__init_subclass__(**kwargs)
        if any(
            getattr(method, "__isabstractmethod__", False)
            for method in (cls._state, cls.from_dict)
        ):
            return
        missing = sorted(
            name
            for name in (
                "__metadata_key__",
                "__format_version__",
                "__state_keys__",
            )
            if not hasattr(cls, name)
        )
        if missing:
            raise TypeError(
                f"{cls.__name__} serializes itself to a file but does not "
                f"declare {', '.join(missing)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this object into a JSON-safe state dictionary.

        The mapping holds the output of :meth:`_state` plus the format version
        under ``"format_version"`` and the class name under ``"__class__"``.
        Arrays may be embedded as ``__ndarray__`` nodes, which the
        serialization layer stores as binary tensors.

        :return: A JSON-safe description suitable for :meth:`from_dict`.
        """
        return self._stamp(self._state())

    @abc.abstractmethod
    def _state(self) -> dict[str, Any]:
        """
        Describes this object as a JSON-safe mapping.

        :return: A JSON-safe mapping holding at least :attr:`__state_keys__`.
        """
        raise NotImplementedError

    def _stamp(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Adds the format version and the class name to a state.

        :param state: A mapping produced by :meth:`_state`.
        :return: The state, headed by ``"format_version"`` and ``"__class__"``.
        """
        return {
            "format_version": self.__format_version__,
            "__class__": type(self).__name__,
            **state,
        }

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls: type[_SerializableT], state: dict[str, Any], **kwargs: Any
    ) -> _SerializableT:
        """
        Rebuilds an object from a state dictionary (see :meth:`to_dict` to see
        the expected format).

        :param state: A JSON-safe description of the object.
        :param kwargs: Objects the state cannot describe, forwarded by
            :meth:`load_from_disk`. Implementations that accept none raise
            an error if ``kwargs`` is not empty.
        :return: A ready-to-use instance.
        """
        raise NotImplementedError

    @classmethod
    def _reject_unsupported_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Raises a :class:`TypeError` if ``kwargs`` is not empty."""
        if kwargs:
            names = ", ".join(repr(name) for name in sorted(kwargs))
            raise TypeError(
                f"{cls.__name__} does not take the deserialization argument(s) {names}."
            )

    def save_to_disk(self, path: str | pathlib.Path) -> pathlib.Path:
        """
        Saves the serialized state of this object to a file.

        :param path: Target file path. Overwritten if it exists.
        :return: The path of the written file.
        :raises OSError: If the destination directory does not exist.
        """
        # The destination is resolved first so that an unwritable one is
        # reported before the state is serialized.
        target = self._resolve_save_path(path)
        return self._write_state(self.to_dict(), target)

    @classmethod
    def load_from_disk(
        cls: type[_SerializableT],
        path: str | pathlib.Path,
        **kwargs: Any,
    ) -> _SerializableT:
        """
        Loads an object previously saved with :meth:`save_to_disk`.

        Not every part of an object survives serialization: an arbitrary
        callable such as a torchvision transform has no portable description,
        so it is left out of the file. Pass such an object back here as a
        keyword argument.

        :param path: Path to the file to load.
        :param kwargs: Objects the file cannot hold, forwarded to
            :meth:`from_dict`.
        :return: A ready-to-use instance.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises ValueError: If the file is not a valid file of this kind or
            was saved by a different class.
        :raises TypeError: If the class does not take one of ``kwargs``.
        """
        file_path = pathlib.Path(path)
        state = cls._read_state(file_path)
        cls._validate_state(state, file_path)
        return cls.from_dict(state, **kwargs)

    @classmethod
    def _resolve_save_path(cls, path: str | pathlib.Path) -> pathlib.Path:
        """
        Turns a target path into the one :meth:`save_to_disk` writes to.

        A destination this library cannot write to is rejected here rather than
        by the safetensors writer.

        :param path: Target file path as given by the caller.
        :return: The path :meth:`save_to_disk` writes to.
        :raises OSError: If the destination directory does not exist.
        """
        path = pathlib.Path(path)
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")
        return path

    def _write_state(self, state: dict[str, Any], path: pathlib.Path) -> pathlib.Path:
        """
        Writes a serialized state to an already-resolved path.

        :param state: A JSON-safe description of the object.
        :param path: Destination path, as returned by :meth:`_resolve_save_path`.
        :return: The path of the written file.
        """
        save_state(state, path, self.__metadata_key__)
        return path

    @classmethod
    def _read_state(cls, path: pathlib.Path) -> dict[str, Any]:
        """
        Reads the serialized state a file holds.

        :param path: Path to the file to read.
        :return: The state, with arrays restored to ``numpy.ndarray``.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises ValueError: If the file cannot be read as a file of this kind.
        """
        if not path.exists():
            raise FileNotFoundError(f"No such file: {str(path)!r}.")
        try:
            return load_state(path, cls.__metadata_key__)
        except ValueError as error:
            raise ValueError(
                f"File {path} is not a valid {cls.__name__} file: {error}"
            ) from error

    @classmethod
    def _validate_state(cls, state: dict[str, Any], path: pathlib.Path) -> None:
        """
        Rejects a state that this class cannot rebuild itself from.

        :param state: The state read from ``path``.
        :param path: Path the state was read from, named in the error messages.
        :raises ValueError: If the state lacks the format version, the class
            name or one of :attr:`__state_keys__`, or was written by another
            class.
        """
        required = cls.__state_keys__ | {"format_version", "__class__"}
        if not required.issubset(state):
            raise ValueError(f"File {path} is not a valid {cls.__name__} file.")
        # TODO: in the future, verify the file's format version against
        # :attr:`__compatibility_mapping__` before reconstructing.
        if state["__class__"] != cls.__name__:
            raise ValueError(
                f"File {path} was saved by {state['__class__']}. "
                f"Load it with {state['__class__']}.load_from_disk instead."
            )
