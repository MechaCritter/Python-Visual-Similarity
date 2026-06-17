from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Iterator, Mapping

import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..typing import Encoder, FloatNumpyArray

#: On-disk format version, bumped if the HDF5 layout ever changes.
_FORMAT_VERSION = 1


class ImageEncodingMap(Mapping[str, FloatNumpyArray]):
    """Map an image path to its encoding vector.

    The encoding for a given path is computed lazily on first access. Once a
    vector has been computed (or loaded) it is kept in an in-memory buffer for
    the rest of the object's lifetime. Call :meth:`clear_buffer` to release the
    buffered vectors; they are recomputed lazily on the next access.

    The full ``path -> encoding`` mapping can be written to and restored from
    an HDF5 file via :meth:`save_to_disk` and :meth:`load_from_disk`.

    :param encoder: Encoder used to turn an image into a fixed-size vector. Any
        object satisfying :class:`pyvisim.typing.Encoder` is accepted.
    :param image_paths: Iterable of image file paths. Duplicates are dropped,
        keeping the first occurrence. May be ``None`` to start empty.
    :raises TypeError: If any provided path is not a string.
    """

    def __init__(
        self,
        encoder: Encoder,
        image_paths: Iterable[str] | None = None,
    ) -> None:
        self.encoder = encoder
        self._paths: list[str] = []
        self._known: set[str] = set()
        self._buffer: dict[str, FloatNumpyArray] = {}

        if image_paths is not None:
            self._register_paths(image_paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._known

    def __getitem__(self, key: str) -> FloatNumpyArray:
        if not isinstance(key, str):
            raise KeyError(
                f"Image key must be a path string, got {type(key).__name__}."
            )
        if key not in self._known:
            raise KeyError(f"Unknown image path: {key!r}.")

        if key not in self._buffer:
            self._buffer[key] = self._encode_path(key)
        return self._buffer[key]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_images={len(self)}, "
            f"encoder={self.encoder.__class__.__name__}, "
            f"buffered={len(self._buffer)})"
        )

    def save_to_disk(self, file_path: str, *, skip_errors: bool = False) -> None:
        """Encode every image and persist the mapping to an HDF5 file.

        :param file_path: Destination ``.h5`` path. Overwritten if it exists.
        :param skip_errors: If ``True``, unreadable images are skipped with a
            warning instead of aborting the whole save.
        :raises ValueError: If the store is empty, no image could be encoded,
            encodings have inconsistent lengths, or (with
            ``skip_errors=False``) an image fails to encode.
        :raises OSError: If the destination directory does not exist.
        """
        if not self._paths:
            raise ValueError("Cannot save an empty ImageStore.")

        parent = os.path.dirname(os.path.abspath(file_path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")

        saved_paths, encodings, failures = self._collect_encodings(skip_errors)

        if failures:
            warnings.warn(
                f"Skipped {len(failures)} image(s) that could not be encoded.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not encodings:
            raise ValueError("No images could be encoded; nothing to save.")
        if len({vector.shape[0] for vector in encodings}) != 1:
            raise ValueError("All encodings must share the same length to be saved.")

        matrix = np.stack(encodings)
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("encodings", data=matrix, compression="gzip")
            handle.create_dataset(
                "paths",
                data=np.array(saved_paths, dtype=object),
                dtype=string_dtype,
            )
            handle.attrs["encoder"] = self.encoder.__class__.__name__
            handle.attrs["format_version"] = _FORMAT_VERSION

    @classmethod
    def load_from_disk(
        cls,
        file_path: str,
        encoder: Encoder,
    ) -> ImageEncodingMap:
        """Rebuild an :class:`ImageEncodingMap` from a :meth:`save_to_disk` file.

        **NOTE**: this loads the encodings straight into the in-memory buffer.

        :param file_path: Path to an HDF5 file produced by :meth:`save_to_disk`.
        :param encoder: Encoder to attach. Re-encoding (e.g. after adding new
            paths) relies on it, so it should match the one used when saving.
        :returns: A populated :class:`ImageEncodingMap`.
        :raises FileNotFoundError: If ``file_path`` does not exist.
        :raises ValueError: If the file is missing the required datasets.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such HDF5 file: {file_path!r}.")

        with h5py.File(file_path, "r") as handle:
            for name in ("encodings", "paths"):
                if name not in handle:
                    raise ValueError(
                        f"{file_path!r} is missing the {name!r} dataset; it "
                        "was not written by ImageEncodingMap.save_to_disk."
                    )
            encodings = handle["encodings"][...]
            paths = list(handle["paths"].asstr()[...])
            stored_encoder = handle.attrs.get("encoder")

        if stored_encoder and stored_encoder != encoder.__class__.__name__:
            warnings.warn(
                f"Encoder mismatch: file saved with {stored_encoder!r}, "
                f"loading with {encoder.__class__.__name__!r}.",
                RuntimeWarning,
                stacklevel=2,
            )

        store = cls(encoder, image_paths=paths)
        for path, vector in zip(paths, encodings, strict=True):
            store._buffer[path] = np.asarray(vector)
        return store

    def clear_buffer(self) -> None:
        """Drop every buffered encoding, freeing the memory they occupy.

        Registered paths are kept, so subsequent access re-encodes the
        corresponding images lazily.
        """
        self._buffer.clear()

    def _register_paths(self, image_paths: Iterable[str]) -> None:
        """Add paths, dropping duplicates and validating their type."""
        for path in image_paths:
            if not isinstance(path, str):
                raise TypeError(
                    f"Image paths must be strings, got {type(path).__name__}."
                )
            if path in self._known:
                continue
            self._known.add(path)
            self._paths.append(path)

    def _encode_path(self, path: str) -> FloatNumpyArray:
        """Open one image and return its flattened encoding."""
        try:
            with Image.open(path) as image:
                rgb_image = np.asarray(image.convert("RGB"))
        except FileNotFoundError:
            raise  # already clear and specific; let it propagate
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read image {path!r}: {exc}") from exc
        return self.encoder.encode(rgb_image).flatten()

    def _collect_encodings(
        self, skip_errors: bool
    ) -> tuple[list[str], list[FloatNumpyArray], list[str]]:
        """Encode all registered paths in preparation for saving.

        :returns: A ``(saved_paths, encodings, failures)`` tuple.
        """
        saved_paths: list[str] = []
        encodings: list[FloatNumpyArray] = []
        failures: list[str] = []
        for path in self._paths:
            try:
                vector = self[path]
            except (KeyError, ValueError, OSError) as exc:
                if not skip_errors:
                    raise ValueError(f"Failed to encode {path!r}: {exc}") from exc
                failures.append(path)
                continue
            saved_paths.append(path)
            encodings.append(np.asarray(vector))
        return saved_paths, encodings, failures
