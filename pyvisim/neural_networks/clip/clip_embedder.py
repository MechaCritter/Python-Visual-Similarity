"""CLIP image embedder built on pyvisim's own CLIP implementation.

The embedder pairs the image towers implemented in :mod:`._model` with
pretrained safetensors weights downloaded from the Hugging Face Hub (see
:mod:`._registry`), so no third-party CLIP library is needed. Every variant
of open_clip's registry whose image tower is a standard CLIP Vision
Transformer or modified ResNet -- including all original OpenAI models -- is
supported.

:mod:`torch` and :mod:`torchvision` are optional dependencies installed by
the ``nn`` extra (``pip install "pyvisim[nn]"``).
"""

from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from ..._base_classes import SimilarityMetric
from ..._utils import get_similarity_func
from ...lazy_import import OptionalImport
from ...typing import (
    Float32NumpyArray,
    FloatNumpyArray,
    ImageInput,
    SimilarityFunc,
    UInt8NumpyArray,
)
from ...utils.image_utils import iter_images
from ._registry import (
    CheckpointSpec,
    VisionConfig,
    fetch_checkpoint,
    get_checkpoint_spec,
    get_model_config,
)

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

    from ._model import build_vision_model, load_vision_weights

_torch_import.check()


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


def _build_preprocess(
    config: VisionConfig, spec: CheckpointSpec
) -> "transforms.Compose":
    """
    Build the CLIP inference preprocessing pipeline of a checkpoint.

    Matches the transform open_clip builds for the checkpoint: a bicubic
    resize (of the shortest side plus a center crop, or of both sides for
    ``"squash"`` checkpoints), conversion to a ``[0, 1]`` tensor and
    normalization with the checkpoint's channel statistics.

    :param config: Architecture description carrying the input size.
    :param spec: Checkpoint metadata carrying the normalization statistics
        and resize mode.
    :return: The composed torchvision transform.
    """
    size = config.image_size
    if spec.resize_mode == "squash":
        resize: list[transforms.Compose | object] = [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            )
        ]
    else:
        resize = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
        ]
    return transforms.Compose(
        [*resize, transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    )


class ClipEmbedder(SimilarityMetric):
    """
    Embeds images with a pretrained CLIP model.

    The safetensors checkpoint of the requested ``variant`` and
    ``pretrained`` tag is downloaded from the Hugging Face Hub on first use
    and cached (see :func:`pyvisim.neural_networks.clip.fetch_checkpoint`);
    only the image tower is loaded, and it always runs in ``float32``.
    :meth:`embed` returns one embedding per image; embeddings are
    L2-normalized by default so they can be compared directly with a dot
    product or the cosine similarity metric.

    Variant names and pretrained tags follow open_clip, e.g.
    ``ClipEmbedder("ViT-B-32", pretrained="laion2b_s34b_b79k")``. OpenAI-style
    variant spellings (``"ViT-B/32"``) are accepted as aliases. See
    :func:`pyvisim.neural_networks.clip.available_variants` and
    :func:`pyvisim.neural_networks.clip.available_pretrained` for the
    supported combinations.

    :param variant: CLIP variant name (default ``"ViT-B-32"``).
    :param pretrained: Pretrained tag naming the weights (default
        ``"openai"``, the original OpenAI checkpoint of the variant).
    :param device: Device to run the model on (``"cpu"`` or ``"cuda"``).
        Defaults to ``"cuda"`` when a CUDA device is available, else
        ``"cpu"``. The model runs in ``float32`` on either device.
    :param normalize: Whether to L2-normalize the returned embeddings
        (default ``True``).
    :param similarity_func: Name of the built-in similarity metric used to
        score two embeddings. One of ``"cosine"`` (default), ``"euclidean"``,
        ``"l1"`` or ``"manhattan"``.
    :param cache_dir: Directory of the Hugging Face Hub cache the checkpoint
        is stored in. Defaults to the standard Hub cache
        (``~/.cache/huggingface/hub``), so weights already downloaded via
        open_clip's Hub downloads are reused.
    :raises ValueError: If ``variant``, ``pretrained`` or
        ``similarity_func`` is not supported, or the checkpoint does not
        match the architecture.
    :raises ImportError: If the ``nn`` extra is not installed.
    :raises huggingface_hub.errors.HfHubHTTPError: If the checkpoint
        download fails.

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
        variant: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str | None = None,
        normalize: bool = True,
        similarity_func: str = "cosine",
        cache_dir: str | Path | None = None,
    ) -> None:
        self._config = get_model_config(variant)
        self._spec = get_checkpoint_spec(variant, pretrained)
        self._variant = variant.replace("/", "-")
        self._pretrained = pretrained
        self._normalize = normalize
        self._device = _resolve_device(device)
        self._similarity_func: SimilarityFunc
        self._similarity_func_name: str
        self.similarity_func = similarity_func
        checkpoint_path = fetch_checkpoint(variant, pretrained, cache_dir=cache_dir)
        model = build_vision_model(self._config, quick_gelu=self._spec.quick_gelu)
        load_vision_weights(model, checkpoint_path)
        self._model = model.eval().to(self._device)
        self._transform = _build_preprocess(self._config, self._spec)

    @property
    def variant(self) -> str:
        """The CLIP variant name in open_clip spelling (e.g. ``"ViT-B-32"``)."""
        return self._variant

    @property
    def pretrained(self) -> str:
        """The pretrained tag naming the loaded weights (e.g. ``"openai"``)."""
        return self._pretrained

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
        return self._config.embed_dim

    @property
    def image_size(self) -> int:
        """Side length in pixels of the square model input."""
        return self._config.image_size

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
        features = self._model(batch)
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
            f"pretrained={self._pretrained}, device={self._device}, "
            f"normalize={self._normalize}, "
            f"similarity_func={self._similarity_func_name})"
        )
