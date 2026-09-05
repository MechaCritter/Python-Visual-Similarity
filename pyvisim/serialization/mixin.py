"""Shared save/load contract for the objects pyvisim writes to a safetensors file."""

from __future__ import annotations

import abc
import os
import pathlib
from typing import Any, ClassVar, TypeVar

from .serialization import load_state, save_state

_SerializableT = TypeVar("_SerializableT", bound="SerializerMixin")


class SerializerMixin(abc.ABC):
    """
    Mixin giving a self-describing object a safetensors file format.

    A subclass declares the kind of file it reads and writes through
    :attr:`_FILE_SUFFIX`, :attr:`_METADATA_KEY` and :attr:`_CLASS_KEY`, and
    describes itself through :meth:`to_dict` / :meth:`from_dict`. In exchange
    it gets :meth:`save_to_disk` and :meth:`load_from_disk`, which are always
    `safetensors <https://github.com/huggingface/safetensors>`_ files: every
    NumPy array of the state is written as a binary tensor, the rest as a
    single JSON blob in the file's metadata.

    The steps in between are overridable hooks:
    :meth:`_resolve_save_path` and :meth:`_write_state` on the way out,
    :meth:`_read_state` and :meth:`_validate_state` on the way in.
    """

    #: Suffix of the files this class writes, appended to a save path when missing.
    _FILE_SUFFIX: ClassVar[str]
    #: Metadata key under which the state's JSON skeleton is stored in the file.
    _METADATA_KEY: ClassVar[str]
    #: State key naming the class that wrote the file.
    _CLASS_KEY: ClassVar[str]
    #: Keys a serialised state must contain to be a valid file of this kind.
    _STATE_KEYS: ClassVar[frozenset[str]] = frozenset()

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialises this object into a JSON-safe state dictionary.

        The returned mapping has to contain at least the keys listed in
        :attr:`_STATE_KEYS`, :attr:`_CLASS_KEY` among them. Arrays may be
        embedded as ``__ndarray__`` nodes; the serialization layer stores them
        as binary tensors.

        :return: A JSON-safe description suitable for :meth:`from_dict`.
        """
        raise NotImplementedError

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
        Saves the serialised state of this object to a file.

        :param path: Target file path. The suffix of the file kind is appended
            if missing. Overwritten if it exists.
        :return: The path of the written file.
        :raises OSError: If the destination directory does not exist.
        """
        # The destination is resolved first so that an unwritable one is
        # reported before the state is serialised.
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

        :attr:`_FILE_SUFFIX` is appended to a path that does not carry it, and
        a destination this library cannot write to is rejected here rather than
        by the safetensors writer.

        :param path: Target file path as given by the caller.
        :return: The path :meth:`save_to_disk` writes to.
        :raises OSError: If the destination directory does not exist.
        """
        path = pathlib.Path(path)
        if path.suffix != cls._FILE_SUFFIX:
            path = path.with_name(path.name + cls._FILE_SUFFIX)
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")
        return path

    def _write_state(self, state: dict[str, Any], path: pathlib.Path) -> pathlib.Path:
        """
        Writes a serialised state to an already-resolved path.

        :param state: A JSON-safe description of the object.
        :param path: Destination path, as returned by :meth:`_resolve_save_path`.
        :return: The path of the written file.
        """
        save_state(state, path, self._METADATA_KEY)
        return path

    @classmethod
    def _read_state(cls, path: pathlib.Path) -> dict[str, Any]:
        """
        Reads the serialised state a file holds.

        :param path: Path to the file to read.
        :return: The state, with arrays restored to ``numpy.ndarray``.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises ValueError: If the file cannot be read as a file of this kind.
        """
        if not path.exists():
            raise FileNotFoundError(f"No such {cls._FILE_SUFFIX} file: {str(path)!r}.")
        return load_state(path, cls._METADATA_KEY)

    @classmethod
    def _validate_state(cls, state: dict[str, Any], path: pathlib.Path) -> None:
        """
        Rejects a state that this class cannot rebuild itself from.

        :param state: The state read from ``path``.
        :param path: Path the state was read from, named in the error messages.
        :raises ValueError: If the state lacks one of :attr:`_STATE_KEYS`, or
            was written by another class.
        """
        if not cls._STATE_KEYS.issubset(state):
            raise ValueError(f"File {path} is not a valid {cls._FILE_SUFFIX} file.")
        # TODO: in the future, verify the file's format version against the
        # class-specific compatibility table before reconstructing.
        if state[cls._CLASS_KEY] != cls.__name__:
            raise ValueError(
                f"File {path} was saved by {state[cls._CLASS_KEY]}. "
                f"Load it with {state[cls._CLASS_KEY]}.load_from_disk instead."
            )
