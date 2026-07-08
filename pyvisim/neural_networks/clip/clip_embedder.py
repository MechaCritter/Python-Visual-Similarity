"""CLIP image embedder loaded straight from OpenAI's official checkpoints.

The embedder downloads OpenAI's original TorchScript archives (see
:mod:`._checkpoints`), verifies them against their published SHA-256 digests
and loads them with :func:`torch.jit.load`, so no CLIP modelling code or
third-party CLIP library is needed. The TorchScript graph-patching helpers
that retarget the archives to the requested device are adapted from OpenAI's
reference implementation at https://github.com/openai/CLIP (MIT license,
Copyright (c) 2021 OpenAI).

:mod:`torch` and :mod:`torchvision` are optional dependencies installed by
the ``nn`` extra (``pip install "pyvisim[nn]"``).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from ..._base_classes import SimilarityMetric
from ..._utils import get_similarity_func
from ...encoders.utils import iter_images
from ...lazy_import import OptionalImport
from ...typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
    UInt8NumpyArray,
)
from ._checkpoints import fetch_checkpoint, get_checkpoint_spec

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()

#: Channel statistics of the dataset CLIP was trained on, as published by
#: OpenAI and mirrored by open_clip's ``OPENAI_DATASET_MEAN`` / ``_STD``.
_OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

#: TorchScript dtype constant for ``float16`` in ``aten::to`` nodes.
_HALF_DTYPE_CONSTANT = 5


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


def _node_value(node: Any, key: str) -> Any:
    """
    Read an attribute of a TorchScript graph node, dispatching on its kind.

    :param node: A ``torch._C.Node`` from a TorchScript graph.
    :param key: Name of the node attribute to read.
    :return: The attribute value.
    """
    selector = node.kindOf(key)
    return getattr(node, selector)(key)


def _module_graphs(module: Any) -> list[Any]:
    """
    Collect the TorchScript graphs owned by a module or script method.

    Plain (non-scripted) modules own no graphs and yield an empty list, so
    the patch helpers below are safe to apply to any module tree.

    :param module: A module or script method to inspect.
    :return: The graphs to patch.
    """
    try:
        graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
        graphs = []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)
    return graphs


def _patch_device(model: Any, device: str) -> None:
    """
    Rewrite hard-coded ``cuda`` device constants in the TorchScript graphs.

    OpenAI's checkpoints were traced on a CUDA machine, so their graphs
    contain literal ``cuda`` device constants that break inference on any
    other device.

    :param model: The loaded TorchScript CLIP model.
    :param device: Device string the constants are rewritten to.
    """
    device_holder = torch.jit.trace(  # type: ignore[no-untyped-call]
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        node
        for node in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(node)
    ][-1]

    def patch(module: Any) -> None:
        for graph in _module_graphs(module):
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(
                    _node_value(node, "value")
                ).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch)
    patch(model.encode_image)
    patch(model.encode_text)


def _patch_float(model: Any) -> None:
    """
    Rewrite hard-coded half-precision casts to ``float32`` and cast the model.

    The traced graphs cast activations to ``float16``, which is only suited
    to CUDA execution and loses precision; the embedder runs the model in
    ``float32`` on every device, matching the default precision open_clip
    uses when it builds a model from the ``openai`` checkpoints.

    :param model: The loaded TorchScript CLIP model.
    """
    float_holder = torch.jit.trace(  # type: ignore[no-untyped-call]
        lambda: torch.ones([]).float(), example_inputs=[]
    )
    float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
    float_node = float_input.node()

    def patch(module: Any) -> None:
        for graph in _module_graphs(module):
            for node in graph.findAllNodes("aten::to"):
                inputs = list(node.inputs())
                # The dtype can be the second or third argument of aten::to().
                for index in (1, 2):
                    if _node_value(inputs[index].node(), "value") == (
                        _HALF_DTYPE_CONSTANT
                    ):
                        inputs[index].node().copyAttributes(float_node)

    model.apply(patch)
    patch(model.encode_image)
    patch(model.encode_text)
    model.float()


def _load_jit_model(path: Path, device: str) -> "torch.jit.ScriptModule":
    """
    Load an OpenAI TorchScript CLIP archive and prepare it for a device.

    The archive is loaded onto the CPU, its graphs are retargeted to
    ``device`` and its baked-in half-precision casts are patched to
    ``float32`` before the weights are converted and moved. The model runs
    in ``float32`` on every device, so embeddings are precise and
    reproducible and match what open_clip produces at its default precision.

    :param path: Path of the verified TorchScript archive.
    :param device: Device the model is prepared for (``"cpu"`` or ``"cuda"``).
    :return: The ready-to-use model in eval mode.
    """
    model = torch.jit.load(  # type: ignore[no-untyped-call]
        str(path), map_location="cpu"
    ).eval()
    _patch_device(model, device)
    _patch_float(model)
    return cast("torch.jit.ScriptModule", model.to(device))


def _build_preprocess(image_size: int) -> "transforms.Compose":
    """
    Build the CLIP inference preprocessing pipeline.

    Matches the transform OpenAI's reference implementation and open_clip use
    for the ``openai`` checkpoints: bicubic resize of the shortest side to
    ``image_size``, center crop, conversion to a ``[0, 1]`` tensor and
    normalization with the CLIP training-set channel statistics.

    :param image_size: Side length in pixels of the square model input.
    :return: The composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_OPENAI_DATASET_MEAN, std=_OPENAI_DATASET_STD),
        ]
    )


class ClipEmbedder(SimilarityMetric):
    """
    Embeds images with an official pretrained OpenAI CLIP model.

    The TorchScript checkpoint of the requested ``variant`` is downloaded
    from OpenAI's public bucket on first use, cached, verified against its
    published SHA-256 digest and loaded with :func:`torch.jit.load` (see
    :func:`pyvisim.neural_networks.clip.fetch_checkpoint`). :meth:`embed`
    runs images through the image tower of the model and returns one
    embedding per image; embeddings are L2-normalized by default so they can
    be compared directly with a dot product or the cosine similarity metric.

    :param variant: Official OpenAI CLIP variant name. One of ``"ViT-B/32"``
        (default), ``"ViT-B/16"`` or ``"ViT-L/14"``.
    :param device: Device to run the model on (``"cpu"`` or ``"cuda"``).
        Defaults to ``"cuda"`` when a CUDA device is available, else
        ``"cpu"``. The model runs in ``float32`` on either device.
    :param normalize: Whether to L2-normalize the returned embeddings
        (default ``True``).
    :param similarity_func: Name of the built-in similarity metric used to
        score two embeddings. One of ``"cosine"`` (default), ``"euclidean"``,
        ``"l1"`` or ``"manhattan"``.
    :param cache_dir: Directory the checkpoint is cached in. Defaults to
        ``~/.cache/clip``, the directory also used by ``open_clip`` and
        OpenAI's ``clip`` package, so existing downloads are reused.
    :raises ValueError: If ``variant`` or ``similarity_func`` is not
        supported.
    :raises pyvisim._errors.ChecksumMismatchError: If the downloaded
        checkpoint fails its SHA-256 integrity check.
    :raises requests.RequestException: If the checkpoint download fails.

    References:
    ===========
    [1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
        Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin,
        Jack Clark, Gretchen Krueger, and Ilya Sutskever, "Learning
        Transferable Visual Models From Natural Language Supervision," in
        Proc. ICML, PMLR 139, pp. 8748-8763, 2021.
    """

    def __init__(
        self,
        variant: str = "ViT-B/32",
        *,
        device: str | None = None,
        normalize: bool = True,
        similarity_func: str = "cosine",
        cache_dir: str | Path | None = None,
    ) -> None:
        self._spec = get_checkpoint_spec(variant)
        self._variant = variant
        self._normalize = normalize
        self._device = _resolve_device(device)
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self.similarity_func = similarity_func
        checkpoint_path = fetch_checkpoint(variant, cache_dir=cache_dir)
        self._model = _load_jit_model(checkpoint_path, self._device)
        self._transform = _build_preprocess(self._spec.image_size)

    @property
    def variant(self) -> str:
        """The official OpenAI CLIP variant name (e.g. ``"ViT-B/32"``)."""
        return self._variant

    @property
    def device(self) -> str:
        """The device the model runs on (``"cpu"`` or ``"cuda"``)."""
        return self._device

    @property
    def normalize(self) -> bool:
        """Whether the returned embeddings are L2-normalized."""
        return self._normalize

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings produced by :meth:`embed`."""
        return self._spec.embedding_dim

    @property
    def image_size(self) -> int:
        """Side length in pixels of the square model input."""
        return self._spec.image_size

    @property
    def similarity_func(self) -> SimilarityFunc:
        """The resolved similarity function callable."""
        return self._similarity_func

    @similarity_func.setter
    def similarity_func(self, name: str) -> None:
        """
        Resolve and store a built-in similarity metric by name.

        :param name: One of ``"cosine"``, ``"euclidean"``, ``"l1"`` or
            ``"manhattan"``.
        :raises ValueError: If ``name`` is not a supported similarity metric.
        """
        self._similarity_func = get_similarity_func(name)
        self._similarity_func_name = name

    def _preprocess(self, image: UInt8NumpyArray) -> "torch.Tensor":
        """
        Convert one canonical ``uint8`` image into a model-ready tensor.

        The image is routed through PIL and converted to RGB, so grayscale
        ``(H, W)`` inputs are accepted as well as ``(H, W, C)`` ones.

        :param image: A ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
        :return: A normalized image tensor of shape
            ``(3, image_size, image_size)``.
        """
        pil_image = Image.fromarray(image).convert("RGB")
        return cast(torch.Tensor, self._transform(pil_image))

    def _encode_image(self, batch: "torch.Tensor") -> "torch.Tensor":
        """
        Run a preprocessed batch through the image tower of the model.

        :param batch: Preprocessed image tensor of shape ``(N, 3, H, W)``.
        :return: Raw image embeddings of shape ``(N, embedding_dim)``.
        """
        encode_image = cast(
            Callable[["torch.Tensor"], "torch.Tensor"], self._model.encode_image
        )
        return encode_image(batch)

    @torch.no_grad()
    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Embed one or more images into CLIP embeddings.

        All images are preprocessed, stacked into a single batch and passed
        through the image tower in one forward pass. When ``normalize`` is
        set, the embeddings are L2-normalized.

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
        :return: ``(N, embedding_dim)`` array holding one embedding per input
            image.
        :raises ValueError: If ``images`` contains no image.
        """
        tensors = [
            self._preprocess(image)
            for image in iter_images(images, dims=dims, value_range=value_range)
        ]
        if not tensors:
            raise ValueError("Expected at least one image to embed, got none.")
        batch = torch.stack(tensors).to(self._device)
        features = self._encode_image(batch)
        if self._normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.float().cpu().numpy(), dtype=np.float32)

    def similarity_score(
        self,
        image1: ImageInput,
        image2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        """
        Compute the similarity matrix between two image batches.

        Both inputs are embedded with :meth:`embed` and the resulting
        embedding batches are scored with ``similarity_func``.

        :param image1: First (batch of) image(s) as ``MatLike``.
        :param image2: Second (batch of) image(s) as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
            (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
            width × channels (NumPy/OpenCV single-image layout, **default**);
            ``"CHW"`` is channels × height × width (PyTorch single-image layout);
            ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
            See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in;
            converted into the canonical ``[0, 255]`` range.
        :return: Similarity matrix of shape ``(N, M)`` produced by
            ``similarity_func``.
        """
        embeddings1 = self.embed(image1, dims=dims, value_range=value_range)
        embeddings2 = self.embed(image2, dims=dims, value_range=value_range)
        return np.asarray(self.similarity_func(embeddings1, embeddings2))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(variant={self._variant}, "
            f"device={self._device}, normalize={self._normalize}, "
            f"similarity_func={self._similarity_func_name})"
        )
