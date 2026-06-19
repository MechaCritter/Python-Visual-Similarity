"""
This script defines various feature extractors used for feature-based image encoders
such as VLAD (Vector of Locally Aggregated Descriptors) or Fisher Vectors. It includes
implementations for handcrafted features (e.g., SIFT, RootSIFT), user-defined
custom feature extraction functions, and deep convolutional feature extraction
with optional spatial encoding.
"""

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

from .._base_classes import FeatureExtractorBase
from .._config import setup_logging
from ..typing import (
    Float32NumpyArray,
    MatLike,
    UInt8NumpyArray,
    _to_image_list,
)

setup_logging()


ExtractorCallT = TypeVar("ExtractorCallT", bound=Callable[..., Any])


def grayscale_dims(image: MatLike, dims: str) -> str:
    """
    Drop the channel label from ``dims`` for a single-channel (grayscale) image.

    A grayscale image carries no channel axis, so an array with exactly one
    fewer dimension than a channel-bearing ``dims`` (e.g. a 2-D array with the
    default ``"HWC"``) is treated as single-channel and the ``"C"`` label is
    removed. This keeps the canonical ``(H, W)`` grayscale layout working with
    the channel-bearing default, matching the NumPy-only behaviour the library
    accepted before ``dims`` were introduced.

    :param image: The image whose axis count is inspected.
    :param dims: The requested axis-label string.
    :return: ``dims`` with ``"C"`` removed when ``image`` is single-channel,
        otherwise ``dims`` unchanged.
    """
    normalized = dims.upper()
    if "C" not in normalized:
        return dims
    if np.ndim(image) == len(normalized) - 1:
        return normalized.replace("C", "")
    return dims


def _to_single_image(
    image: MatLike,
    dims: str = "HWC",
    value_range: tuple[float, float] = (0.0, 255.0),
) -> UInt8NumpyArray:
    """
    Normalize a single ``MatLike`` image into one canonical array.

    :param image: A NumPy array, a PyTorch tensor, or any array-like object.
    :param dims: Axis-label string, one character per array axis in order:
        ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels
        (e.g. RGB), ``"B"`` = batch size. For example, ``"HWC"`` is height ×
        width × channels (NumPy/OpenCV single-image layout, **default**);
        ``"CHW"`` is channels × height × width (PyTorch single-image layout);
        ``"BCHW"`` is batch × channels × height × width (PyTorch batched layout).
        A single-channel (grayscale) image may be passed as a 2-D array with the
        default ``"HWC"``; the channel label is dropped automatically.
    :param value_range: The ``(low, high)`` range the input values live in
        (default ``(0.0, 255.0)``).
    :return: A ``uint8`` image of shape ``(H, W[, C])`` in ``[0, 255]``.
    :raises ValueError: If the input expands to anything other than one image.
    """
    images = _to_image_list(image, grayscale_dims(image, dims), value_range)
    if len(images) != 1:
        raise ValueError(
            f"Expected a single image, but the input expands to {len(images)} images. "
            "Feature extractors operate on one image at a time; use an encoder for "
            "batches."
        )
    return images[0]


def _check_output_shape(  # noqa: UP047
    func: ExtractorCallT,
) -> ExtractorCallT:
    """
    Ensures the feature extractor output is a 2D NumPy array of shape
    (num_vectors, self.output_dim).

    Input normalization (``MatLike`` conversion plus ``dims``/``value_range``
    handling) is performed inside each wrapped ``__call__``; this decorator
    only validates the output.
    """

    @wraps(func)
    def wrapper(
        self: FeatureExtractorBase, image: MatLike, /, *args: Any, **kwargs: Any
    ) -> Float32NumpyArray:
        feat_vecs = func(self, image, *args, **kwargs)
        if feat_vecs is None:
            print("No feature vectors found. Returning empty array.")
            return np.zeros((0, self.output_dim), dtype=np.float32)

        if not isinstance(feat_vecs, np.ndarray):
            raise ValueError(
                f"Expected output to be a NumPy array, got {type(feat_vecs)} instead."
            )

        if feat_vecs.ndim != 2:
            raise ValueError(
                f"Feature extractor output must be 2D. Got shape {feat_vecs.shape}."
            )

        if feat_vecs.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected feat_vecs.shape[1] == {self.output_dim}, "
                f"but got {feat_vecs.shape[1]}."
            )

        return feat_vecs

    return cast(ExtractorCallT, wrapper)


class SIFT(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform (SIFT) feature extractor.

    References:
    ===========
    [1] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self._output_dim = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts SIFT features from an image.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: ``(num_keypoints, 128)`` array of SIFT descriptors.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        return cast(Float32NumpyArray, descriptors)

    def __repr__(self) -> str:
        return f"SIFT(output_dim={self.output_dim})"


class RootSIFT(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform with Hellinger kernel (RootSIFT) normalizer.

    References:
    ===========
    [1] Arandjelovic, R., & Zisserman, A. (2012). Three things everyone should know to improve object retrieval.
    """

    def __init__(self) -> None:
        super().__init__()
        self._output_dim = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts RootSIFT features from an image.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: ``(num_keypoints, 128)`` array of RootSIFT descriptors.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        sift = cv2.SIFT.create()
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
            return np.asarray(np.sqrt(descriptors))
        return descriptors

    def __repr__(self) -> str:
        return f"RootSIFT(output_dim={self.output_dim})"


class Lambda(FeatureExtractorBase):
    """
    Lambda feature extractor that allows passing any user-defined
    function to extract features from images.

    The function must accept a single argument (image as NumPy array),
    and output fixed-size feature vectors from each image.
    """

    def __init__(
        self, func: Callable[[UInt8NumpyArray], Float32NumpyArray], output_dim: int
    ):
        """
        Initializes the Lambda feature extractor.
        :param func:
        :param output_dim:
        """
        super().__init__()
        if not callable(func):
            raise ValueError(
                f"Argument func must be a callable object, got {type(func)} instead"
            )
        self._output_dim = output_dim
        self.func = func

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _serialization_config(self) -> dict[str, Any]:
        """
        Lambda extractors wrap an arbitrary user function and cannot be
        serialised.

        :raises TypeError: Always; pass ``feature_extractor`` explicitly when
            loading an encoder that used a Lambda extractor.
        """
        raise TypeError(
            "Lambda feature extractors wrap a user-defined function and cannot "
            "be serialised. Provide 'feature_extractor' explicitly when loading."
        )

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Extracts features from an image using the user-defined function.

        :param image: Input image as ``MatLike``.
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: Feature descriptors returned by ``func``.
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        return self.func(image)


class DeepConvFeature(FeatureExtractorBase):
    """
    Extracts convolutional feature maps from a chosen conv layer of a torchvision model.
    It flattens the feature maps into feature descriptors. Optionally appends
    normalized (x, y) coordinates to each spatial location.

    The concepts here were inspired by by the work on `VLAD-DCNN` features for face verification, as
    presented in [1], where VLAD encodings were computed from deep convolutional features and input into
    a metric learning algorithm in order to distinguish between different people.

    :param backbone: The convolutional backbone to extract features from. It may be:

        * ``None`` (default): builds a torchvision VGG16 with ImageNet weights.
        * A string naming a built-in backbone. Currently only ``"vgg16"`` is
          supported, which builds a torchvision VGG16 with ImageNet weights.
        * A ``torch.nn.Module`` instance: any user-supplied PyTorch model.

        In the paper [1], a VGG-Face model trained on the Imdb-Wiki dataset was
        used with VLAD encoding for younger faces verification.
    :param target_submodule: Optional submodule name to hook into. If None, the whole model is used.
    :param layer_index: Which conv layer to hook (int). Use `list_conv_layers(...)`
                       to see the ordering or use -1 for the last conv layer.
    :param spatial_encoding: If True, appends (x/W, y/H) to each descriptor.
    :param device: 'cpu' or 'cuda'. Where to run the model.
    :param transform: Optional torchvision.transforms.Compose. Default includes `to_tensor`, `resize(224, 224)`,
                        and normalization with ImageNet stats.

    .. deprecated:: 0.4.1
        The ``model`` keyword argument is deprecated; pass the model through
        ``backbone`` instead. When ``model`` is supplied it is used as the
        ``backbone`` (unless ``backbone`` is also given) and a
        :class:`FutureWarning` is emitted.

    References:
    ===========
    [1] Liangliang Wang and Deepu Rajan, "An Image Similarity Descriptor for Classification Tasks," J. Vis. Commun. Image R., vol. 71, pp. 102847, 2020.
    [2] Weixia Zhang, Jia Yan, Wenxuan Shi, Tianpeng Feng, and Dexiang Deng, "Refining Deep Convolutional Features for Improving Fine-Grained Image
    Recognition," EURASIP Journal on Image and Video Processing, 2017.
    """

    def __init__(
        self,
        backbone: str | torch.nn.Module | None = None,
        target_submodule: str | None = None,
        layer_index: int = -1,
        spatial_encoding: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        transform: transforms.Compose = None,
        **kwargs: Any,
    ):
        super().__init__()
        backbone = self._resolve_deprecated_model(backbone, kwargs)
        # Track whether a rebuildable (torchvision-default) model is used: when
        # serialising, such a model is rebuilt from torchvision on load (no
        # weights stored), while a user-supplied model has its state_dict
        # embedded in the encoder file.
        model, self._is_default_model = _build_backbone(backbone)
        self._model: torch.nn.Module
        self._target_submodule = target_submodule
        self.layer_index = layer_index
        self.spatial_encoding = spatial_encoding
        self.device = device
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            )

        self.model: torch.nn.Module = model  # Trigger setter
        self._modules: torch.nn.Module = self._get_submodule(target_submodule)
        self._conv_layers = self.list_conv_layers()
        if not self._conv_layers:
            raise ValueError(
                f"No convolutional layers found in model {type(self.model).__name__}."
            )

        self.buffer: torch.Tensor | None = None
        try:
            _, self.selected_layer_name, self.selected_layer_module = self._conv_layers[
                self.layer_index
            ]
            self._logger.info(
                f"Selected layer: {self.selected_layer_name}, {self.selected_layer_module}"
            )
        except IndexError as e:
            info = (
                ""
                if target_submodule is None
                else f" in submodule {type(self._modules).__name__}"
            )
            raise IndexError(
                f"Model {type(self.model).__name__} has only {len(self._conv_layers)} convolutional layers {info}"
                f". Got layer_index={self.layer_index}."
            ) from e
        self._output_dim = (
            self.selected_layer_module.out_channels + 2
            if self.spatial_encoding
            else self.selected_layer_module.out_channels
        )
        self._register_hook()

    @staticmethod
    def _resolve_deprecated_model(
        backbone: str | torch.nn.Module | None,
        kwargs: dict[str, Any],
    ) -> str | torch.nn.Module | None:
        """
        Resolve the deprecated ``model`` keyword argument into ``backbone``.

        If ``model`` is present in ``kwargs`` a :class:`FutureWarning` is
        emitted. The popped ``model`` is used as the ``backbone`` only when no
        explicit ``backbone`` was supplied; otherwise ``backbone`` wins.

        :param backbone: The ``backbone`` argument as passed by the caller.
        :param kwargs: Extra keyword arguments captured by ``__init__``.
        :return: The backbone to use.
        :raises TypeError: If ``kwargs`` contains unexpected keyword arguments.
        """
        if "model" in kwargs:
            warnings.warn(
                "The 'model' argument of DeepConvFeature is deprecated and will "
                "be removed in a future release; pass the model through "
                "'backbone' instead.",
                FutureWarning,
                stacklevel=3,
            )
            model = kwargs.pop("model")
            if backbone is None:
                backbone = model
        if kwargs:
            raise TypeError(
                f"DeepConvFeature got unexpected keyword arguments: {sorted(kwargs)}."
            )
        return backbone

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _serialization_config(self) -> dict[str, Any]:
        """
        Return the configuration needed to rebuild this deep feature extractor.

        When the default model is used (``model=None`` at construction), only
        its architecture name is stored and the model is rebuilt from
        torchvision's default weights on load. When a user-supplied model is
        used, its full ``state_dict`` is embedded (as array nodes that the
        encoder serializer writes as binary tensors) so the trained weights
        are recovered exactly. The custom ``transform`` is not serialised; the
        default transform is used when reconstructing.

        :return: A mapping of constructor arguments. It may contain encoded
            array nodes (the model ``state_dict``) which the encoder serializer
            extracts into the ``.encoder`` file's tensors.
        """
        config: dict[str, Any] = {
            "model_arch": type(self._model).__name__,
            "default_model": self._is_default_model,
            "target_submodule": self._target_submodule,
            "layer_index": self.layer_index,
            "spatial_encoding": self.spatial_encoding,
            "device": self.device,
        }
        if not self._is_default_model:
            config["state_dict"] = _encode_state_dict(self._model)
        return config

    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> "DeepConvFeature":
        """
        Rebuild a :class:`DeepConvFeature` from a serialised configuration.

        For a default model the architecture is rebuilt from torchvision's
        default weights. For a user-supplied model the architecture skeleton is
        rebuilt and the embedded ``state_dict`` is loaded into it, recovering
        the trained weights exactly.

        :param config: Mapping produced by :meth:`_serialization_config`.
        :return: A reconstructed deep feature extractor.
        :raises ValueError: If the stored model architecture is not known and
            cannot be rebuilt automatically.
        """
        arch = config["model_arch"]
        default_model = config.get("default_model", True)
        device = config.get("device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = _build_torchvision_model(arch, pretrained=default_model)
        if not default_model:
            model.load_state_dict(_decode_state_dict(config["state_dict"]))
        return cls(
            backbone=model,
            target_submodule=config.get("target_submodule"),
            layer_index=config["layer_index"],
            spatial_encoding=config["spatial_encoding"],
            device=device,
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Currently, only torch.nn.Module is supported. Got {type(model)} instead."
            )
        self._model = model

    def _get_submodule(self, submodule_name: str | None = None) -> torch.nn.Module:
        """
        Retrieves a submodule from a PyTorch model by name.

        :return: The submodule instance.
        """
        if submodule_name is None:
            return self._model
        if not hasattr(self._model, submodule_name):
            raise AttributeError(
                f"Model {type(self.model).__name__} has no submodule named {submodule_name}."
            )
        submodule = getattr(self._model, submodule_name)
        if not isinstance(submodule, torch.nn.Module):
            raise TypeError(
                f"Attribute {submodule_name} of model {type(self.model).__name__} "
                f"is not a torch.nn.Module, got {type(submodule)} instead."
            )
        return submodule

    def list_conv_layers(self) -> list[tuple[int, str, torch.nn.Conv2d]]:
        """
        Utility function to collect convolutional layers (and sub-modules)
        from the model / chosen submodule.

        :return: List of (layer_index, layer_module) for each convolutional layer.
        """
        conv_layers: list[tuple[int, str, torch.nn.Conv2d]] = []
        idx = 0
        for name, module in self._modules.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((idx, name, module))
                idx += 1
        return conv_layers

    def _register_hook(self) -> None:
        """
        Registers a forward hook on the selected convolutional layer
        to capture its output (feature map).
        """

        def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> None:
            self.buffer = (
                output.detach()
            )  # output shape: [batch_size, channels, height, width]

        self.hook = self.selected_layer_module.register_forward_hook(hook_fn)

    @_check_output_shape
    def __call__(
        self,
        image: MatLike,
        /,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> Float32NumpyArray:
        """
        Processes a single image through the chosen conv layer and
        returns flattened feature descriptors.

        The input is normalized to a canonical ``uint8`` ``(H, W, C)`` image
        and then passed through ``self.transform`` (which converts it to a
        tensor in ``[0, 1]``). Batches are handled at the encoder level, so a
        single image is expected here.

        :param image: Input image as ``MatLike`` (e.g. a NumPy ``(H, W, C)``
            array or a torch ``(C, H, W)`` tensor; pass ``dims`` accordingly).
        :param dims: Axis-label string, one character per array axis in order:
            ``"H"`` = height (rows), ``"W"`` = width (columns), ``"C"`` = channels.
            For example, ``"HWC"`` is height × width × channels (NumPy/OpenCV
            layout, **default**); ``"CHW"`` is channels × height × width (PyTorch
            layout). See :mod:`pyvisim.typing`.
        :param value_range: The ``(low, high)`` range the input values live in.
        :return: N x D NumPy array, where N = (H_conv x W_conv) and
                 D = number_of_channels (+ 2 if spatial coords are appended).
        """
        image = _to_single_image(image, dims=dims, value_range=value_range)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        self.model.to(self.device)
        _ = self.model(input_tensor)  # we only care about the hook's output
        if self.buffer is None:
            raise RuntimeError("Forward hook did not capture any features.")

        # Convert the captured feature map to NumPy
        feature_map = self.buffer.cpu().numpy()  # shape: (1, C, Hf, Wf)
        feature_map = feature_map[0]  # Remove batch dimension

        C, Hf, Wf = feature_map.shape
        feature_map = feature_map.reshape(C, -1).T  # shape: (Hf*Wf, C)

        if self.spatial_encoding:
            coords = []
            for y in range(Hf):
                for x in range(Wf):
                    coords.append([x / Wf, y / Hf])  # (x/Wf, y/Hf)
            coords_array = np.array(coords, dtype=np.float32)  # shape: (Hf*Wf, 2)
            # Concatenate
            feature_map = np.hstack([feature_map, coords_array])  # shape: (Hf*Wf, C+2)

        return feature_map

    def __repr__(self) -> str:
        return (
            f"DeepConvFeature(backbone={type(self.model).__name__}, layer_index={self.layer_index}, "
            f"spatial_encoding={self.spatial_encoding}, device={self.device}, "
            f"transform={self.transform}, selected_layer_name={self.selected_layer_name}, "
            f"selected_layer_module={self.selected_layer_module}, output_dim={self.output_dim})"
        )


#: Maps a backbone identifier to a builder. The builder takes a ``pretrained``
#: flag: ``True`` loads torchvision's default (ImageNet) weights, ``False``
#: returns the bare architecture so embedded weights can be loaded into it.
#:
#: Each backbone is registered under two identifiers so a single map serves both
#: lookups: the friendly name users pass via ``backbone`` (e.g. ``"vgg16"``) and
#: the model class name serialization recovers from a model instance
#: (``type(model).__name__``, e.g. ``"VGG"``). Currently only VGG16 is supported.
_BACKBONE_BUILDERS: dict[str, Callable[[bool], torch.nn.Module]] = {
    "vgg16": lambda pretrained: vgg16(
        weights=VGG16_Weights.DEFAULT if pretrained else None
    ),
}
_BACKBONE_BUILDERS["VGG"] = _BACKBONE_BUILDERS["vgg16"]


def _build_backbone(
    backbone: str | torch.nn.Module | None,
) -> tuple[torch.nn.Module, bool]:
    """
    Resolve a ``backbone`` argument into a concrete model.

    :param backbone: ``None`` for the default VGG16, a string naming a built-in
        backbone (see :data:`_BACKBONE_BUILDERS`), or a ``torch.nn.Module``.
    :return: A ``(model, is_default_model)`` tuple. ``is_default_model`` is
        ``True`` when the model can be rebuilt from torchvision's default
        weights on load (``None`` or a built-in name) and ``False`` for a
        user-supplied module whose weights must be embedded when serialising.
    :raises ValueError: If ``backbone`` is an unknown built-in name.
    :raises TypeError: If ``backbone`` is neither ``None``, a string, nor a
        ``torch.nn.Module``.
    """
    if backbone is None:
        return _BACKBONE_BUILDERS["vgg16"](True), True
    if isinstance(backbone, str):
        builder = _BACKBONE_BUILDERS.get(backbone.lower())
        if builder is None:
            # Only the friendly, lowercase names are public; the class-name
            # aliases exist purely for serialization round-trips.
            raise ValueError(
                f"Unknown backbone {backbone!r}. Supported backbones: "
                f"{sorted(name for name in _BACKBONE_BUILDERS if name.islower())}."
            )
        return builder(True), True
    if isinstance(backbone, torch.nn.Module):
        return backbone, False
    raise TypeError(
        "backbone must be None, a string naming a built-in backbone, or a "
        f"torch.nn.Module. Got {type(backbone)} instead."
    )


def _build_torchvision_model(arch: str, *, pretrained: bool) -> torch.nn.Module:
    """
    Rebuild a torchvision model from its architecture name.

    :param arch: Model architecture name (e.g. ``"VGG"``).
    :param pretrained: Whether to load the default (ImageNet) weights.
    :return: The reconstructed model.
    :raises ValueError: If the architecture is not known.
    """
    builder = _BACKBONE_BUILDERS.get(arch)
    if builder is None:
        raise ValueError(
            f"Cannot automatically rebuild model architecture {arch!r}. "
            "Provide 'feature_extractor' explicitly when loading."
        )
    return builder(pretrained)


def _encode_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """
    Encode a model's ``state_dict`` as array nodes for the encoder serializer.

    :param model: The model whose weights are serialised.
    :return: A mapping of parameter name to an encoded array node. The encoder
        serializer extracts these arrays into the ``.encoder`` file's tensors.
    """
    encoded: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()
        encoded[name] = {
            "__ndarray__": True,
            "data": array,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "order": "C",
        }
    return encoded


def _decode_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Rebuild a model ``state_dict`` from arrays restored by the encoder loader.

    :param state_dict: Mapping of parameter name to NumPy array (restored from
        the ``.encoder`` file's tensors).
    :return: A mapping of parameter name to torch tensor.
    """
    return {
        name: torch.from_numpy(np.ascontiguousarray(array))
        for name, array in state_dict.items()
    }


#: Stateless feature extractors that take no constructor arguments.
_STATELESS_FEATURE_EXTRACTORS: dict[str, type[FeatureExtractorBase]] = {
    "SIFT": SIFT,
    "RootSIFT": RootSIFT,
}


def feature_extractor_from_dict(data: dict[str, Any]) -> FeatureExtractorBase:
    """
    Rebuild a feature extractor from a dict produced by
    :meth:`FeatureExtractorBase.to_dict`.

    :param data: Mapping ``{"__class__": str, "config": dict}``.
    :return: A reconstructed feature extractor instance.
    :raises TypeError: If ``data`` is not a mapping with the expected keys.
    :raises ValueError: If the extractor class cannot be reconstructed
        automatically (e.g. a Lambda extractor).
    """
    if not isinstance(data, dict) or "__class__" not in data:
        raise TypeError("Expected a feature-extractor dict from to_dict().")
    name = data["__class__"]
    config = data.get("config", {})
    if name in _STATELESS_FEATURE_EXTRACTORS:
        return _STATELESS_FEATURE_EXTRACTORS[name]()
    if name == "DeepConvFeature":
        return DeepConvFeature._from_config(config)
    raise ValueError(
        f"Cannot reconstruct feature extractor {name!r} automatically. "
        "Provide 'feature_extractor' explicitly when loading."
    )
