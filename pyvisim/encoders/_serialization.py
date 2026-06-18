"""
safetensors-backed serialization for image encoders.

An encoder is described by a nested, JSON-safe state dictionary (clustering
model, optional PCA, normalization hyperparameters, similarity-metric name and
feature-extractor configuration). This module stores that description as a
``.encoder`` file in the `safetensors <https://github.com/huggingface/safetensors>`_
format: every NumPy array is written as a binary tensor, while the surrounding
structure and scalar values are stored as a single JSON blob in the file's
metadata.
"""

import json
import pathlib
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from ..typing import NumpyArray

#: Metadata key under which the JSON skeleton is stored in the safetensors file.
_METADATA_KEY = "pyvisim_encoder"


def _arrays_to_tensors(
    obj: Any, tensors: dict[str, NumpyArray], counter: list[int]
) -> Any:
    """
    Replace encoded ``__ndarray__`` nodes with tensor references.

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


def save_encoder_state(state: dict[str, Any], path: pathlib.Path) -> None:
    """
    Write an encoder state dictionary to a ``.encoder`` safetensors file.

    :param state: JSON-safe encoder description (may contain ``__ndarray__``
        nodes produced by the clustering models' ``to_dict``).
    :param path: Destination file path.
    """
    tensors: dict[str, NumpyArray] = {}
    skeleton = _arrays_to_tensors(state, tensors, [0])
    save_file(tensors, str(path), metadata={_METADATA_KEY: json.dumps(skeleton)})


def load_encoder_state(path: pathlib.Path) -> dict[str, Any]:
    """
    Read an encoder state dictionary from a ``.encoder`` safetensors file.

    :param path: Path to the ``.encoder`` file.
    :return: The reconstructed encoder state, with arrays restored.
    :raises ValueError: If the file is not a valid ``.encoder`` file.
    """
    try:
        with safe_open(str(path), framework="numpy") as handle:
            metadata = handle.metadata() or {}
            raw_skeleton = metadata.get(_METADATA_KEY)
            if raw_skeleton is None:
                raise ValueError(f"File {path} is not a valid .encoder file.")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            skeleton = json.loads(raw_skeleton)
    except (SafetensorError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"File {path} is not a valid .encoder file.") from error
    state = _tensors_to_arrays(skeleton, tensors)
    if not isinstance(state, dict):
        raise ValueError(f"File {path} is not a valid .encoder file.")
    return state
