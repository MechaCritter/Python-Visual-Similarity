"""Minimal reader for MATLAB Level-5 MAT-files.

This module implements just enough of the MAT-file format to load the numeric
arrays shipped with the Oxford 102 Flowers dataset (``imagelabels.mat`` and
``setid.mat``), which lets us drop the runtime dependency on SciPy. Only
numeric and character arrays are supported; cell, struct, object and sparse
classes raise :class:`ValueError`.

Reference: *MAT-File Format*, R2018b, The MathWorks, Inc.
"""

import struct
import zlib
from typing import Any

import numpy as np
import numpy.typing as npt

_NDArray = npt.NDArray[Any]

#: Size in bytes of the fixed MAT-file header.
_HEADER_SIZE = 128

#: MAT-file data-type tags (``miXXX``) mapped to their NumPy dtype character.
_NUMERIC_DTYPES: dict[int, str] = {
    1: "i1",  # miINT8
    2: "u1",  # miUINT8
    3: "i2",  # miINT16
    4: "u2",  # miUINT16
    5: "i4",  # miINT32
    6: "u4",  # miUINT32
    7: "f4",  # miSINGLE
    9: "f8",  # miDOUBLE
    12: "i8",  # miINT64
    13: "u8",  # miUINT64
    16: "u1",  # miUTF8
    17: "u2",  # miUTF16
    18: "u4",  # miUTF32
}

#: Data-type tag of a matrix element.
_MI_MATRIX = 14
#: Data-type tag of a zlib-compressed element.
_MI_COMPRESSED = 15

#: Array classes (``mxXXX_CLASS``) we can decode: character and every numeric
#: class. Cell (1), struct (2), object (3) and sparse (5) are rejected.
_SUPPORTED_CLASSES = frozenset({4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})


def _read_endianness(buffer: bytes) -> str:
    """
    Determine the byte order of a MAT-file from its header.

    :param buffer: Raw contents of the MAT-file.
    :return: The :mod:`struct` byte-order prefix (``"<"`` or ``">"``).
    :raises ValueError: If the header is missing or the endian indicator is
        unrecognised.
    """
    if len(buffer) < _HEADER_SIZE:
        raise ValueError("File is too small to be a valid MAT-file.")
    indicator = buffer[126:128]
    if indicator == b"IM":
        return "<"
    if indicator == b"MI":
        return ">"
    raise ValueError("Unrecognised endian indicator; not a Level-5 MAT-file.")


def _read_tag(buffer: bytes, offset: int, endian: str) -> tuple[int, int, int, int]:
    """
    Read a data-element tag, transparently handling the small-element format.

    :param buffer: Buffer to read from.
    :param offset: Offset of the tag within ``buffer``.
    :param endian: Byte-order prefix for :mod:`struct`.
    :return: Tuple of ``(data_type, data_offset, num_bytes, next_offset)`` where
        ``data_offset`` is the start of the element payload and ``next_offset``
        points at the following element (padded to an 8-byte boundary).
    """
    (first,) = struct.unpack_from(endian + "I", buffer, offset)
    small_num_bytes = first >> 16
    if small_num_bytes:
        data_type = first & 0xFFFF
        return data_type, offset + 4, small_num_bytes, offset + 8
    (num_bytes,) = struct.unpack_from(endian + "I", buffer, offset + 4)
    data_offset = offset + 8
    padding = (8 - num_bytes % 8) % 8
    return first, data_offset, num_bytes, data_offset + num_bytes + padding


def _read_numeric(
    buffer: bytes, offset: int, endian: str, dims: tuple[int, ...]
) -> _NDArray:
    """
    Read a numeric data element and reshape it to the given dimensions.

    :param buffer: Buffer to read from.
    :param offset: Offset of the data element within ``buffer``.
    :param endian: Byte-order prefix for :mod:`struct` and NumPy.
    :param dims: Column-major array dimensions.
    :return: The decoded array.
    :raises ValueError: If the element uses an unsupported data type.
    """
    data_type, data_offset, num_bytes, _ = _read_tag(buffer, offset, endian)
    try:
        dtype_char = _NUMERIC_DTYPES[data_type]
    except KeyError:
        raise ValueError(f"Unsupported MAT data type {data_type}.") from None
    raw = bytes(buffer[data_offset : data_offset + num_bytes])
    array = np.frombuffer(raw, dtype=np.dtype(endian + dtype_char))
    return array.reshape(dims, order="F")


def _parse_matrix(buffer: bytes, offset: int, endian: str) -> tuple[str, _NDArray]:
    """
    Parse the body of a ``miMATRIX`` element into its name and array.

    :param buffer: Buffer holding the matrix body.
    :param offset: Offset of the array-flags subelement.
    :param endian: Byte-order prefix for :mod:`struct` and NumPy.
    :return: Tuple of the variable name and its array.
    :raises ValueError: If the array class is not numeric or character.
    """
    _, flags_offset, _, offset = _read_tag(buffer, offset, endian)
    (flags_word,) = struct.unpack_from(endian + "I", buffer, flags_offset)
    array_class = flags_word & 0xFF
    if array_class not in _SUPPORTED_CLASSES:
        raise ValueError(f"Unsupported MAT array class {array_class}.")

    _, dims_offset, dims_bytes, offset = _read_tag(buffer, offset, endian)
    ndims = dims_bytes // 4
    dims = struct.unpack_from(endian + f"{ndims}i", buffer, dims_offset)

    _, name_offset, name_bytes, offset = _read_tag(buffer, offset, endian)
    name = bytes(buffer[name_offset : name_offset + name_bytes]).decode("ascii")

    array = _read_numeric(buffer, offset, endian, dims)
    return name, array


def _read_element(
    buffer: bytes, offset: int, endian: str
) -> tuple[str | None, _NDArray | None, int]:
    """
    Read a single top-level data element, decompressing it if needed.

    :param buffer: Buffer to read from.
    :param offset: Offset of the element within ``buffer``.
    :param endian: Byte-order prefix for :mod:`struct` and NumPy.
    :return: Tuple of ``(name, array, next_offset)``. ``name`` and ``array`` are
        ``None`` for elements that are not matrices.
    """
    data_type, data_offset, num_bytes, next_offset = _read_tag(buffer, offset, endian)
    if data_type == _MI_COMPRESSED:
        # Compressed elements are written back-to-back and, unlike other
        # elements, are not padded to a 64-bit boundary.
        next_offset = data_offset + num_bytes
        payload = zlib.decompress(bytes(buffer[data_offset : data_offset + num_bytes]))
        inner_type, inner_offset, _, _ = _read_tag(payload, 0, endian)
        if inner_type != _MI_MATRIX:
            return None, None, next_offset
        name, array = _parse_matrix(payload, inner_offset, endian)
        return name, array, next_offset
    if data_type == _MI_MATRIX:
        name, array = _parse_matrix(buffer, data_offset, endian)
        return name, array, next_offset
    return None, None, next_offset


def load_mat(path: str) -> dict[str, _NDArray]:
    """
    Load the numeric variables from a MATLAB Level-5 MAT-file.

    :param path: Path to the ``.mat`` file.
    :return: Mapping of variable name to its array.
    :raises ValueError: If the file is not a valid Level-5 MAT-file or contains
        an unsupported array class or data type.
    """
    with open(path, "rb") as handle:
        buffer = handle.read()

    endian = _read_endianness(buffer)
    variables: dict[str, _NDArray] = {}
    offset = _HEADER_SIZE
    while offset + 8 <= len(buffer):
        name, array, offset = _read_element(buffer, offset, endian)
        if name is not None and array is not None:
            variables[name] = array
    return variables
