"""
Shared base for the neural image embedders.
"""

import abc
from typing import Any

from .._base_classes import ImageEmbedderBase
from ..lazy_import import OptionalImport
from ..typing import FloatNumpyArray, ImageInput

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch

_torch_import.check()


class NeuralImageEmbedder(ImageEmbedderBase, torch.nn.Module):
    """
    Abstract base for image embedders backed by a torch module.

    It combines the two halves every neural embedder in pyvisim needs:
    :class:`~pyvisim._base_classes.ImageEmbedderBase` contributes the
    ``similarity_func`` handling and the :meth:`similarity_score` built on top
    of :meth:`embed`, while :class:`torch.nn.Module` contributes the parameter,
    device and training-mode machinery.

    Unlike the classic embedders, neural ones are not serialised to
    ``.embedder`` files; persist them with the usual torch checkpoints
    (``state_dict``).

    Subclasses implement :meth:`embed` and must call this constructor before
    registering any submodule, so that ``torch.nn.Module`` is initialized first.

    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    """

    def __init__(self, similarity_func: str = "cosine") -> None:
        # torch.nn.Module.__init__ creates the attribute dicts that its
        # __setattr__ relies on, so it has to run before anything is assigned.
        torch.nn.Module.__init__(self)
        ImageEmbedderBase.__init__(self, similarity_func=similarity_func)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute, routing property assignments through the property.

        NOTE
        ----
        Since :class:`torch.nn.Module` overrides ``__setattr__`` and registers
        any ``torch.nn.Module`` value directly in ``self._modules``, assigning
        to a read-only property such as ``head`` would silently register an
        orphan submodule instead of failing. Routing assignments to class-level
        properties through ``property.__set__`` restores standard Python
        semantics. Hence, this override was necessary.

        :param name: Name of the attribute to set.
        :param value: Value to assign.
        :raises AttributeError: If ``name`` is a read-only property.
        """
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property):
            descriptor.__set__(self, value)
            return
        super().__setattr__(name, value)

    @abc.abstractmethod
    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        raise NotImplementedError

    @property
    def device(self) -> "torch.device":
        """
        The device the model's parameters live on.

        Derived from the parameters themselves rather than cached, so it stays
        correct after the user moves the model with ``model.to(...)``.
        """
        return next(self.parameters()).device
