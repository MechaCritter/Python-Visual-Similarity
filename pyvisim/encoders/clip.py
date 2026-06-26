"""CLIP image encoder built on top of the `open_clip <https://github.com/mlfoundations/open_clip>`_ library.

:mod:`open_clip` is an optional dependency installed by the ``nn`` extra
(``pip install "pyvisim[nn]"``). It is imported lazily through
:class:`~pyvisim.lazy_import.OptionalImport`, so importing :mod:`pyvisim` never
requires it: the actionable :class:`ImportError` is raised only when a
:class:`CLIPEncoder` is constructed without the dependency installed.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image

from .._config import setup_logging
from ..lazy_import import OptionalImport
from ..typing import Float32NumpyArray, ImageInput, UInt8NumpyArray
from ._base_encoder import ImageEncoderBase
from .utils import iter_images

with OptionalImport(package="open_clip_torch", extra="nn") as _open_clip_import:
    import open_clip

setup_logging()

#: Bumped whenever the serialised :class:`CLIPEncoder` state layout changes.
_CLIP_ENCODER_FILE_FORMAT_VERSION = 1


def _resolve_device(device: str | None) -> str:
    """
    Resolve a requested device, falling back to CPU when CUDA is unavailable.

    :param device: ``"cuda"``, ``"cpu"`` or ``None`` to auto-select.
    :return: ``"cuda"`` if requested/available, otherwise ``"cpu"``.
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


class CLIPEncoder(ImageEncoderBase):
    """
    Encodes images into CLIP embeddings using a pretrained `open_clip` model.

    The model and its matching preprocessing transform are loaded from
    `open_clip` at construction time. :meth:`encode` runs each image through the
    image tower of the model and returns its embedding; embeddings are
    L2-normalized by default so they can be compared directly with a dot product
    or the cosine similarity metric.

    Because the weights are pretrained and downloaded by `open_clip`,
    serialization stores only the identifiers needed to rebuild the encoder
    (``model_name`` and ``pretrained`` tag) rather than the weights themselves.

    :param model_name: `open_clip` architecture name, e.g. ``"ViT-B-32"``. See
        https://pypi.org/project/open-clip-torch/ for all available models.
    :param pretrained: `open_clip` pretrained weight tag, e.g.
        ``"laion2b_s34b_b79k"``. Run ``open_clip.list_pretrained()`` to see the
        available tags. ``None`` builds the architecture with randomly
        initialized weights (no download).
    :param device: Device to run the model on (``"cpu"`` or ``"cuda"``).
        Defaults to ``"cuda"`` when a CUDA device is available, else ``"cpu"``.
    :param normalize: Whether to L2-normalize the returned embeddings
        (default ``True``).
    :param similarity_func: Name of the built-in similarity metric to use. One of
        ``"cosine"`` (default), ``"euclidean"``, ``"l1"`` or ``"manhattan"``.
    :raises ImportError: If the optional ``open_clip`` dependency is not installed.

    References:
    ===========
    [1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
        Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
        Gretchen Krueger, and Ilya Sutskever, "Learning Transferable Visual Models
        From Natural Language Supervision," in Proc. ICML, PMLR 139, pp. 8748–8763,
        2021.

    [2] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas
        Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong,
        John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt,
        "OpenCLIP," Zenodo, Jul. 2021.
        doi: 10.5281/zenodo.5143773
    """

    #: Keys a serialised state must contain to be a valid encoder file.
    _STATE_KEYS = frozenset(
        {
            "encoder_class",
            "similarity_func",
            "model_name",
            "pretrained",
            "normalize",
        }
    )

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str | None = "laion2b_s34b_b79k",
        *,
        device: str | None = None,
        normalize: bool = True,
        similarity_func: str = "cosine",
    ) -> None:
        _open_clip_import.check()
        super().__init__(similarity_func=similarity_func)
        self._model_name = model_name
        self._pretrained = pretrained
        self._normalize = normalize
        self._device = _resolve_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    @property
    def model_name(self) -> str:
        """The `open_clip` architecture name (e.g. ``"ViT-B-32"``)."""
        return self._model_name

    @property
    def pretrained(self) -> str | None:
        """The `open_clip` pretrained weight tag (e.g. ``"laion2b_s34b_b79k"``)."""
        return self._pretrained

    @property
    def device(self) -> str:
        """The device the model runs on (``"cpu"`` or ``"cuda"``)."""
        return self._device

    @property
    def normalize(self) -> bool:
        """Whether the returned embeddings are L2-normalized."""
        return self._normalize

    @staticmethod
    def _to_pil(image: UInt8NumpyArray) -> Image.Image:
        """
        Convert a canonical ``uint8`` image array into a PIL image.

        The `open_clip` preprocessing transform converts the image to RGB, so a
        grayscale ``(H, W)`` array is accepted as well as an ``(H, W, C)`` one.

        :param image: A ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
        :return: The corresponding :class:`PIL.Image.Image`.
        """
        return Image.fromarray(image)

    def encode(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Encode one or more images into CLIP embeddings.

        Each image is normalized to a canonical ``uint8`` ``(H, W, C)`` array,
        converted to a PIL image, passed through the `open_clip` preprocessing
        transform and the image tower of the model. When ``normalize`` is set,
        the embeddings are L2-normalized.

        :param images: A single ``MatLike`` image, a batched array, or an
            iterable of images. Consider using an iterator for large datasets.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: ``(N, D)`` array holding one ``D``-dimensional embedding per
            input image.
        """
        embeddings: list[Float32NumpyArray] = []
        for image in iter_images(images, dims=dims, value_range=value_range):
            tensor = self._preprocess(self._to_pil(image)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
            if self._normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(
                np.asarray(features.detach().cpu().numpy(), dtype=np.float32)
            )
        return np.vstack(embeddings)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise this encoder into a JSON-safe state dictionary.

        Only the identifiers needed to rebuild the encoder are stored; the
        pretrained weights are re-downloaded by `open_clip` on :meth:`from_dict`.

        :return: A JSON-safe encoder description suitable for :meth:`from_dict`.
        """
        return {
            "format_version": _CLIP_ENCODER_FILE_FORMAT_VERSION,
            "encoder_class": type(self).__name__,
            "model_name": self._model_name,
            "pretrained": self._pretrained,
            "normalize": self._normalize,
            "device": self._device,
            "similarity_func": self._similarity_func_name,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "CLIPEncoder":
        """
        Rebuild a :class:`CLIPEncoder` from a dictionary produced by :meth:`to_dict`.

        :param state: A JSON-safe encoder description from :meth:`to_dict`.
        :return: A ready-to-use encoder instance.
        """
        return cls(
            model_name=state["model_name"],
            pretrained=state["pretrained"],
            device=state.get("device"),
            normalize=state["normalize"],
            similarity_func=state["similarity_func"],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_name={self._model_name}, "
            f"pretrained={self._pretrained}, device={self._device}, "
            f"normalize={self._normalize}, "
            f"similarity_func={self.similarity_func_name})"
        )
