"""
Helpers shared by the torch-backed parts of pyvisim.

The functions here bridge :mod:`torch` and the array-based serialization
layer (:mod:`pyvisim.serialization`): a model's ``state_dict`` is turned
into embedded array nodes that are written as binary tensors of a
``.embedder`` file, and back.

:mod:`torch` is an optional dependency installed by the ``nn`` extra
(``pip install "pyvisim[nn]"``); importing this module requires it.
"""

from typing import Any

import numpy as np

from ..lazy_import import OptionalImport
from ..serialization import decode_array_node

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch

_torch_import.check()


def resolve_device(device: "str | torch.device | None") -> str:
    """
    Resolve a requested device, falling back to CPU when CUDA is unavailable.

    :param device: A device name or :class:`torch.device` (e.g. ``"cuda"``,
        ``"cuda:0"``, ``"cpu"``), or ``None`` to auto-select.
    :return: The name of the device to use; ``"cuda"`` if requested or
        available, otherwise ``"cpu"``.
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    name = str(device)
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def encode_state_dict(module: "torch.nn.Module") -> dict[str, Any]:
    """
    Embed a module's ``state_dict`` as array nodes for the embedder serializer.

    :param module: The module whose weights are serialised.
    :return: A mapping of parameter name to an embedded array node. The embedder
        serializer extracts these arrays into the ``.embedder`` file's tensors.
    """
    embedded: dict[str, Any] = {}
    for name, tensor in module.state_dict().items():
        array = tensor.detach().cpu().numpy()
        embedded[name] = {
            "__ndarray__": True,
            "data": array,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "order": "C",
        }
    return embedded


def decode_state_dict(state_dict: dict[str, Any]) -> dict[str, "torch.Tensor"]:
    """
    Rebuild a module ``state_dict`` from arrays restored by the embedder loader.

    :param state_dict: Mapping of parameter name to a NumPy array (restored
        from the ``.embedder`` file's tensors) or to the ``__ndarray__`` node
        :func:`encode_state_dict` produced.
    :return: A mapping of parameter name to torch tensor.
    """
    return {
        name: torch.from_numpy(_writable_array(decode_array_node(value)))
        for name, value in state_dict.items()
    }


def _writable_array(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Return a C-contiguous, writable view (or copy) of ``array``.

    Arrays read back from a safetensors file may be backed by read-only
    memory, which :func:`torch.from_numpy` cannot wrap without warning.

    :param array: The array restored from the ``.embedder`` file.
    :return: An equivalent array that torch can wrap safely.
    """
    array = np.ascontiguousarray(array)
    return array if array.flags.writeable else array.copy()
