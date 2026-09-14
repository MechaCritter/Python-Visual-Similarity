"""safetensors-backed serialization for image embedders."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from ..typing import NumpyArray


def decode_array_node(value: Any) -> NumpyArray:
    """
    Restore the array held by an ``__ndarray__`` node.

    Values that already are arrays (because they were restored from the
    file's binary tensors) are returned as such, so callers can consume a
    state dictionary regardless of whether it went through a file.

    :param value: An ``__ndarray__`` node or an array-like value.
    :return: The array the node describes.
    """
    if isinstance(value, dict) and value.get("__ndarray__"):
        array = np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
        return np.asfortranarray(array) if value.get("order") == "F" else array
    return np.asarray(value)


def _arrays_to_tensors(
    obj: Any, tensors: dict[str, NumpyArray], counter: list[int]
) -> Any:
    """
    Replace embedded ``__ndarray__`` nodes with tensor references.

    Walks a JSON-safe structure (as produced by the clustering models'
    ``to_dict``) and moves every array into ``tensors`` under a unique key,
    leaving a ``{"__tensor__": key, "order": ...}`` placeholder behind.

    :param obj: The structure to walk.
    :param tensors: Accumulator mapping tensor keys to NumPy arrays.
    :param counter: Single-element list used as a mutable key counter.
    :return: The structure with arrays replaced by tensor references.
    """
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            key = f"tensor_{counter[0]}"
            counter[0] += 1
            array = np.asarray(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            tensors[key] = np.ascontiguousarray(array)
            return {"__tensor__": key, "order": obj.get("order", "C")}
        return {
            key: _arrays_to_tensors(value, tensors, counter)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_arrays_to_tensors(value, tensors, counter) for value in obj]
    return obj


def _tensors_to_arrays(obj: Any, tensors: dict[str, NumpyArray]) -> Any:
    """
    Restore tensor references back into NumPy arrays.

    Inverse of :func:`_arrays_to_tensors`: every ``{"__tensor__": key}``
    placeholder is replaced by the corresponding array, re-applying
    Fortran memory order when it was recorded.

    :param obj: The structure to walk.
    :param tensors: Mapping of tensor keys to NumPy arrays.
    :return: The structure with tensor references replaced by arrays.
    """
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            array = tensors[obj["__tensor__"]]
            if obj.get("order") == "F":
                array = np.asfortranarray(array)
            return array
        return {key: _tensors_to_arrays(value, tensors) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_tensors_to_arrays(value, tensors) for value in obj]
    return obj


def save_state(
    state: dict[str, Any], path: str | pathlib.Path, metadata_key: str
) -> None:
    """
    Write a JSON-safe state dictionary to a safetensors file.

    Every ``__ndarray__`` node found anywhere in ``state`` (as produced by the
    clustering models' ``to_dict``) is extracted into a binary tensor, while the
    surrounding structure is stored as a single JSON blob under ``metadata_key``.

    :param state: JSON-safe description (may contain ``__ndarray__`` nodes).
    :param path: Destination file path.
    :param metadata_key: Metadata key under which the JSON skeleton is stored.
    """
    tensors: dict[str, NumpyArray] = {}
    skeleton = _arrays_to_tensors(state, tensors, [0])
    save_file(tensors, str(path), metadata={metadata_key: json.dumps(skeleton)})


def load_state(path: str | pathlib.Path, metadata_key: str) -> dict[str, Any]:
    """
    Read a JSON-safe state dictionary written by :func:`save_state`.

    :param path: Path to the safetensors file.
    :param metadata_key: Metadata key the JSON skeleton was stored under.
    :return: The reconstructed state, with arrays restored to ``numpy.ndarray``.
    :raises ValueError: If the file cannot be read or lacks ``metadata_key``.
    """
    try:
        with safe_open(str(path), framework="numpy") as handle:
            metadata = handle.metadata() or {}
            raw_skeleton = metadata.get(metadata_key)
            if raw_skeleton is None:
                raise ValueError(f"File {path} is missing the {metadata_key!r} key.")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            skeleton = json.loads(raw_skeleton)
    except (SafetensorError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"File {path} could not be read as a pyvisim file.") from error
    state = _tensors_to_arrays(skeleton, tensors)
    if not isinstance(state, dict):
        raise ValueError(f"File {path} does not hold a state dictionary.")
    return state
