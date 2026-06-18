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

    The full ``path -> encoding`` mapping can be written to and restored from
    an HDF5 file via :meth:`save_to_disk` and :meth:`load_from_disk`.

    :param encoder: Encoder used to turn an image into a fixed-size vector. Any
        object satisfying :class:`pyvisim.typing.Encoder` is accepted.
    :param image_paths: Iterable of image file paths. Duplicates are dropped,
        keeping the first occurrence. May be ``None`` to start empty.
    :param skip_errors: If ``True``, images that cannot be read or encoded are
        skipped with a warning instead of aborting construction.
    :raises TypeError: If any provided path is not a string.
    :raises FileNotFoundError: If an image file is missing (and
        ``skip_errors`` is ``False``).
    :raises ValueError: If an image cannot be decoded (and ``skip_errors`` is
        ``False``).
    """

    def __init__(
        self,
        encoder: Encoder,
        image_paths: Iterable[str] | None = None,
        *,
        skip_errors: bool = False,
    ) -> None:
        self.encoder = encoder
        self._encodings: dict[str, FloatNumpyArray] = {}

        if image_paths is not None:
            self._encode_paths(image_paths, skip_errors)

    def __len__(self) -> int:
        return len(self._encodings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._encodings)

    def __contains__(self, key: object) -> bool:
        return key in self._encodings

    def __getitem__(self, key: str) -> FloatNumpyArray:
        if not isinstance(key, str):
            raise KeyError(
                f"Image key must be a path string, got {type(key).__name__}."
            )
        if key not in self._encodings:
            raise KeyError(f"Unknown image path: {key!r}.")
        return self._encodings[key]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_images={len(self)}, "
            f"encoder={self.encoder.__class__.__name__})"
        )

    def save_to_disk(self, file_path: str) -> None:
        """Persist the ``path -> encoding`` mapping to an HDF5 file.

        :param file_path: Destination ``.h5`` path. Overwritten if it exists.
        :raises ValueError: If the store is empty or the encodings have
            inconsistent lengths.
        :raises OSError: If the destination directory does not exist.
        """
        if not self._encodings:
            raise ValueError("Cannot save an empty ImageStore.")

        parent = os.path.dirname(os.path.abspath(file_path))
        if not os.path.isdir(parent):
            raise OSError(f"Destination directory does not exist: {parent!r}.")

        saved_paths = list(self._encodings)
        encodings = [np.asarray(vector) for vector in self._encodings.values()]
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

        The encodings are restored directly from the file; no image is
        re-encoded.

        :param file_path: Path to an HDF5 file produced by :meth:`save_to_disk`.
        :param encoder: Encoder to attach. It should match the one used when
            saving.
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

        store = cls(encoder)
        store._encodings = {
            path: np.asarray(vector)
            for path, vector in zip(paths, encodings, strict=True)
        }
        return store

    def _encode_paths(self, image_paths: Iterable[str], skip_errors: bool) -> None:
        """Encode every path, dropping duplicates and validating their type.

        :param image_paths: Iterable of image file paths to encode.
        :param skip_errors: If ``True``, unreadable images are skipped with a
            warning instead of raising.
        """
        failures: list[str] = []
        for path in image_paths:
            if not isinstance(path, str):
                raise TypeError(
                    f"Image paths must be strings, got {type(path).__name__}."
                )
            if path in self._encodings:
                continue
            try:
                self._encodings[path] = self._encode_path(path)
            except (FileNotFoundError, ValueError, OSError):
                if not skip_errors:
                    raise
                failures.append(path)

        if failures:
            warnings.warn(
                f"Skipped {len(failures)} image(s) that could not be encoded.",
                RuntimeWarning,
                stacklevel=2,
            )

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
