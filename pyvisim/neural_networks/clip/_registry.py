"""Registry of pretrained CLIP checkpoints hosted on the Hugging Face Hub.

Checkpoints are downloaded with :func:`huggingface_hub.hf_hub_download` into
the standard Hugging Face cache (``~/.cache/huggingface/hub`` unless
``HF_HOME``/``HF_HUB_CACHE`` says otherwise).
"""

from dataclasses import dataclass
from pathlib import Path

from ...lazy_import import OptionalImport

with OptionalImport(package="huggingface_hub", extra="nn") as _hf_hub_import:
    from huggingface_hub import hf_hub_download

#: Channel statistics of the dataset CLIP was trained on, as published by
#: OpenAI and mirrored by open_clip's ``OPENAI_DATASET_MEAN`` / ``_STD``.
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

#: Channel statistics used by a handful of LAION checkpoints instead.
_INCEPTION_MEAN = (0.5, 0.5, 0.5)
_INCEPTION_STD = (0.5, 0.5, 0.5)

#: Preferred open_clip safetensors file name on the Hub; a few LAION
#: repositories only publish the alternative name below.
_SAFETENSORS_NAME = "open_clip_model.safetensors"
_SAFETENSORS_NAME_ALT = "open_clip_pytorch_model.safetensors"


@dataclass(frozen=True)
class VisionConfig:
    """
    Architecture description of one CLIP image tower.

    :param embed_dim: Dimensionality of the image embedding space.
    :param image_size: Side length in pixels of the square model input.
    :param width: Transformer token width (ViT) or stem width (ResNet).
    :param layers: Number of transformer blocks (ViT) or bottleneck blocks
        per stage (ResNet).
    :param patch_size: Side length in pixels of one square ViT patch;
        ``None`` for the ResNet variants.
    :param head_width: Channel width of one attention head.
    :param mlp_ratio: Ratio between the transformer MLP hidden width and
        ``width``.
    """

    embed_dim: int
    image_size: int
    width: int
    layers: int | tuple[int, int, int, int]
    patch_size: int | None = None
    head_width: int = 64
    mlp_ratio: float = 4.0

    @property
    def heads(self) -> int:
        """Number of attention heads of the transformer or attention pool."""
        if isinstance(self.layers, tuple):
            return self.width * 32 // self.head_width
        return self.width // self.head_width

    @property
    def mlp_width(self) -> int:
        """Hidden width of the transformer MLPs."""
        return int(self.width * self.mlp_ratio)


@dataclass(frozen=True)
class CheckpointSpec:
    """
    Location and preprocessing metadata of one pretrained checkpoint.

    :param repo_id: Hugging Face Hub repository holding the weights.
    :param filename: Name of the safetensors file inside the repository.
    :param quick_gelu: Whether the checkpoint was trained with the QuickGELU
        activation (original OpenAI models and their contemporaries) rather
        than the exact GELU used by newer checkpoints.
    :param mean: Per-channel normalization mean of the preprocessing.
    :param std: Per-channel normalization std of the preprocessing.
    :param resize_mode: ``"shortest"`` resizes the shortest image side and
        center-crops; ``"squash"`` resizes both sides, ignoring aspect ratio.
    """

    repo_id: str
    filename: str = _SAFETENSORS_NAME
    quick_gelu: bool = False
    mean: tuple[float, float, float] = OPENAI_DATASET_MEAN
    std: tuple[float, float, float] = OPENAI_DATASET_STD
    resize_mode: str = "shortest"

    @property
    def url(self) -> str:
        """Hugging Face Hub page of the checkpoint repository."""
        return f"https://huggingface.co/{self.repo_id}"


_RN50_CONFIG = VisionConfig(1024, 224, 64, (3, 4, 6, 3))
_RN101_CONFIG = VisionConfig(512, 224, 64, (3, 4, 23, 3))
_RN50X4_CONFIG = VisionConfig(640, 288, 80, (4, 6, 10, 6))
_RN50X16_CONFIG = VisionConfig(768, 384, 96, (6, 8, 18, 8))
_RN50X64_CONFIG = VisionConfig(1024, 448, 128, (3, 15, 36, 10))
_VITB32_CONFIG = VisionConfig(512, 224, 768, 12, patch_size=32)
_VITB32_256_CONFIG = VisionConfig(512, 256, 768, 12, patch_size=32)
_VITB16_CONFIG = VisionConfig(512, 224, 768, 12, patch_size=16)
_VITB16_PLUS_240_CONFIG = VisionConfig(640, 240, 896, 12, patch_size=16)
_VITL14_CONFIG = VisionConfig(768, 224, 1024, 24, patch_size=14)
_VITL14_336_CONFIG = VisionConfig(768, 336, 1024, 24, patch_size=14)
_VITH14_CONFIG = VisionConfig(1024, 224, 1280, 32, patch_size=14, head_width=80)
_VITH14_378_CONFIG = VisionConfig(1024, 378, 1280, 32, patch_size=14, head_width=80)
_VITG14_CONFIG = VisionConfig(
    1024, 224, 1408, 40, patch_size=14, head_width=88, mlp_ratio=4.3637
)
_VITBIGG14_CONFIG = VisionConfig(
    1280, 224, 1664, 48, patch_size=14, head_width=104, mlp_ratio=4.9231
)
_VITBIGG14_378_CONFIG = VisionConfig(
    1280, 378, 1664, 48, patch_size=14, head_width=104, mlp_ratio=4.9231
)

#: Image-tower architecture of every supported variant name.
MODEL_CONFIGS: dict[str, VisionConfig] = {
    "RN50": _RN50_CONFIG,
    "RN50-quickgelu": _RN50_CONFIG,
    "RN101": _RN101_CONFIG,
    "RN101-quickgelu": _RN101_CONFIG,
    "RN50x4": _RN50X4_CONFIG,
    "RN50x4-quickgelu": _RN50X4_CONFIG,
    "RN50x16": _RN50X16_CONFIG,
    "RN50x16-quickgelu": _RN50X16_CONFIG,
    "RN50x64": _RN50X64_CONFIG,
    "RN50x64-quickgelu": _RN50X64_CONFIG,
    "ViT-B-32": _VITB32_CONFIG,
    "ViT-B-32-quickgelu": _VITB32_CONFIG,
    "ViT-B-32-256": _VITB32_256_CONFIG,
    "ViT-B-16": _VITB16_CONFIG,
    "ViT-B-16-quickgelu": _VITB16_CONFIG,
    "ViT-B-16-plus-240": _VITB16_PLUS_240_CONFIG,
    "ViT-L-14": _VITL14_CONFIG,
    "ViT-L-14-quickgelu": _VITL14_CONFIG,
    "ViT-L-14-336": _VITL14_336_CONFIG,
    "ViT-L-14-336-quickgelu": _VITL14_336_CONFIG,
    "ViT-H-14": _VITH14_CONFIG,
    "ViT-H-14-quickgelu": _VITH14_CONFIG,
    "ViT-H-14-worldwide": _VITH14_CONFIG,
    "ViT-H-14-worldwide-quickgelu": _VITH14_CONFIG,
    "ViT-H-14-worldwide-378": _VITH14_378_CONFIG,
    "ViT-g-14": _VITG14_CONFIG,
    "ViT-bigG-14": _VITBIGG14_CONFIG,
    "ViT-bigG-14-quickgelu": _VITBIGG14_CONFIG,
    "ViT-bigG-14-worldwide": _VITBIGG14_CONFIG,
    "ViT-bigG-14-worldwide-378": _VITBIGG14_378_CONFIG,
}

_RN50_TAGS = {
    "openai": CheckpointSpec("timm/resnet50_clip.openai", quick_gelu=True),
    "yfcc15m": CheckpointSpec("timm/resnet50_clip.yfcc15m", quick_gelu=True),
    "cc12m": CheckpointSpec("timm/resnet50_clip.cc12m", quick_gelu=True),
}
_RN101_TAGS = {
    "openai": CheckpointSpec("timm/resnet101_clip.openai", quick_gelu=True),
    "yfcc15m": CheckpointSpec("timm/resnet101_clip.yfcc15m", quick_gelu=True),
}
_RN50X4_TAGS = {
    "openai": CheckpointSpec("timm/resnet50x4_clip.openai", quick_gelu=True),
}
_RN50X16_TAGS = {
    "openai": CheckpointSpec("timm/resnet50x16_clip.openai", quick_gelu=True),
}
_RN50X64_TAGS = {
    "openai": CheckpointSpec("timm/resnet50x64_clip.openai", quick_gelu=True),
}
_VITL14_336_TAGS = {
    "openai": CheckpointSpec("timm/vit_large_patch14_clip_336.openai", quick_gelu=True),
}
_VITH14_WORLDWIDE_TAGS = {
    "metaclip2_worldwide": CheckpointSpec(
        "timm/vit_huge_patch14_clip_224.metaclip2_worldwide", quick_gelu=True
    ),
}

#: Pretrained tags of every supported variant name. Repositories, activation
#: flags and preprocessing metadata mirror open_clip's pretrained registry.
PRETRAINED: dict[str, dict[str, CheckpointSpec]] = {
    "RN50": _RN50_TAGS,
    "RN50-quickgelu": _RN50_TAGS,
    "RN101": _RN101_TAGS,
    "RN101-quickgelu": _RN101_TAGS,
    "RN50x4": _RN50X4_TAGS,
    "RN50x4-quickgelu": _RN50X4_TAGS,
    "RN50x16": _RN50X16_TAGS,
    "RN50x16-quickgelu": _RN50X16_TAGS,
    "RN50x64": _RN50X64_TAGS,
    "RN50x64-quickgelu": _RN50X64_TAGS,
    "ViT-B-32": {
        "openai": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.openai", quick_gelu=True
        ),
        "laion400m_e31": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.laion400m_e31", quick_gelu=True
        ),
        "laion400m_e32": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.laion400m_e32", quick_gelu=True
        ),
        "laion2b_e16": CheckpointSpec("timm/vit_base_patch32_clip_224.laion2b_e16"),
        "laion2b_s34b_b79k": CheckpointSpec("laion/CLIP-ViT-B-32-laion2B-s34B-b79K"),
        "datacomp_xl_s13b_b90k": CheckpointSpec(
            "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        ),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-B-32-quickgelu": {
        "openai": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.openai", quick_gelu=True
        ),
        "laion400m_e31": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.laion400m_e31", quick_gelu=True
        ),
        "laion400m_e32": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.laion400m_e32", quick_gelu=True
        ),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_base_patch32_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-B-32-256": {
        "datacomp_s34b_b86k": CheckpointSpec(
            "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
        ),
    },
    "ViT-B-16": {
        "openai": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.openai", quick_gelu=True
        ),
        "laion400m_e31": CheckpointSpec("timm/vit_base_patch16_clip_224.laion400m_e31"),
        "laion400m_e32": CheckpointSpec("timm/vit_base_patch16_clip_224.laion400m_e32"),
        "laion2b_s34b_b88k": CheckpointSpec("laion/CLIP-ViT-B-16-laion2B-s34B-b88K"),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-B-16-quickgelu": {
        "openai": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.openai", quick_gelu=True
        ),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_base_patch16_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-B-16-plus-240": {
        "laion400m_e31": CheckpointSpec(
            "timm/vit_base_patch16_plus_clip_240.laion400m_e31"
        ),
        # open_clip's registry maps the e32 tag to the e31 repository too.
        "laion400m_e32": CheckpointSpec(
            "timm/vit_base_patch16_plus_clip_240.laion400m_e31"
        ),
    },
    "ViT-L-14": {
        "openai": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.openai", quick_gelu=True
        ),
        "laion400m_e31": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.laion400m_e31"
        ),
        "laion400m_e32": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.laion400m_e32"
        ),
        "laion2b_s32b_b82k": CheckpointSpec(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            filename=_SAFETENSORS_NAME_ALT,
            mean=_INCEPTION_MEAN,
            std=_INCEPTION_STD,
        ),
        "commonpool_xl_s13b_b90k": CheckpointSpec(
            "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
            filename=_SAFETENSORS_NAME_ALT,
        ),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-L-14-quickgelu": {
        "openai": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.openai", quick_gelu=True
        ),
        "metaclip_400m": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.metaclip_400m", quick_gelu=True
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_large_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-L-14-336": _VITL14_336_TAGS,
    "ViT-L-14-336-quickgelu": _VITL14_336_TAGS,
    "ViT-H-14": {
        "laion2b_s32b_b79k": CheckpointSpec("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_huge_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
        "metaclip_altogether": CheckpointSpec(
            "timm/vit_huge_patch14_clip_224.metaclip_altogether"
        ),
    },
    "ViT-H-14-quickgelu": {
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_huge_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-H-14-worldwide": _VITH14_WORLDWIDE_TAGS,
    "ViT-H-14-worldwide-quickgelu": _VITH14_WORLDWIDE_TAGS,
    "ViT-H-14-worldwide-378": {
        "metaclip2_worldwide": CheckpointSpec(
            "timm/vit_huge_patch14_clip_378.metaclip2_worldwide",
            resize_mode="squash",
        ),
    },
    "ViT-g-14": {
        "laion2b_s12b_b42k": CheckpointSpec(
            "laion/CLIP-ViT-g-14-laion2B-s12B-b42K", filename=_SAFETENSORS_NAME_ALT
        ),
        "laion2b_s34b_b88k": CheckpointSpec("laion/CLIP-ViT-g-14-laion2B-s34B-b88K"),
    },
    "ViT-bigG-14": {
        "laion2b_s39b_b160k": CheckpointSpec(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        ),
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-bigG-14-quickgelu": {
        "metaclip_fullcc": CheckpointSpec(
            "timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b", quick_gelu=True
        ),
    },
    "ViT-bigG-14-worldwide": {
        "metaclip2_worldwide": CheckpointSpec(
            "timm/vit_gigantic_patch14_clip_224.metaclip2_worldwide"
        ),
    },
    "ViT-bigG-14-worldwide-378": {
        "metaclip2_worldwide": CheckpointSpec(
            "timm/vit_gigantic_patch14_clip_378.metaclip2_worldwide",
            resize_mode="squash",
        ),
    },
}


def _canonical_variant(variant: str) -> str:
    """
    Normalize a variant name to its open_clip spelling.

    OpenAI-style names (``"ViT-B/32"``) are accepted and mapped to
    open_clip-style names (``"ViT-B-32"``), like open_clip itself does.

    :param variant: Variant name in either spelling.
    :return: The open_clip-style variant name.
    """
    return variant.replace("/", "-")


def available_variants() -> tuple[str, ...]:
    """
    List the names of the supported CLIP variants.

    :return: The variant names accepted by :func:`get_model_config`.
    """
    return tuple(MODEL_CONFIGS)


def available_pretrained(variant: str) -> tuple[str, ...]:
    """
    List the pretrained tags available for a CLIP variant.

    :param variant: Variant name, e.g. ``"ViT-B-32"``.
    :return: The tags accepted by :func:`get_checkpoint_spec` for it.
    :raises ValueError: If ``variant`` is not a supported variant.
    """
    variant = _canonical_variant(variant)
    if variant not in PRETRAINED:
        raise ValueError(
            f"Unsupported CLIP variant: {variant!r}. "
            f"Supported variants: {', '.join(PRETRAINED)}."
        )
    return tuple(PRETRAINED[variant])


def get_model_config(variant: str) -> VisionConfig:
    """
    Look up the image-tower architecture of a CLIP variant.

    :param variant: Variant name, e.g. ``"ViT-B-32"`` (or ``"ViT-B/32"``).
    :return: The matching :class:`VisionConfig`.
    :raises ValueError: If ``variant`` is not a supported variant.
    """
    variant = _canonical_variant(variant)
    try:
        return MODEL_CONFIGS[variant]
    except KeyError:
        raise ValueError(
            f"Unsupported CLIP variant: {variant!r}. "
            f"Supported variants: {', '.join(MODEL_CONFIGS)}."
        ) from None


def get_checkpoint_spec(variant: str, pretrained: str) -> CheckpointSpec:
    """
    Look up the checkpoint of a CLIP variant and pretrained tag.

    :param variant: Variant name, e.g. ``"ViT-B-32"`` (or ``"ViT-B/32"``).
    :param pretrained: Pretrained tag, e.g. ``"openai"``.
    :return: The matching :class:`CheckpointSpec`.
    :raises ValueError: If ``variant`` or ``pretrained`` is not supported.
    """
    variant = _canonical_variant(variant)
    if variant not in PRETRAINED:
        raise ValueError(
            f"Unsupported CLIP variant: {variant!r}. "
            f"Supported variants: {', '.join(PRETRAINED)}."
        )
    tags = PRETRAINED[variant]
    try:
        return tags[pretrained]
    except KeyError:
        raise ValueError(
            f"Unsupported pretrained tag for {variant!r}: {pretrained!r}. "
            f"Available tags: {', '.join(tags)}."
        ) from None


def fetch_checkpoint(
    variant: str, pretrained: str, cache_dir: str | Path | None = None
) -> Path:
    """
    Return a local path to the safetensors checkpoint of a CLIP variant.

    The file is downloaded from the Hugging Face Hub on first use and read
    from the local Hub cache afterwards; huggingface_hub verifies the
    integrity of every download. Weights already cached by open_clip's Hub
    downloads are reused instead of re-downloaded.

    :param variant: Variant name, e.g. ``"ViT-B-32"`` (or ``"ViT-B/32"``).
    :param pretrained: Pretrained tag, e.g. ``"openai"``.
    :param cache_dir: Directory of the Hub cache. Defaults to the standard
        Hugging Face cache (``~/.cache/huggingface/hub``, honoring
        ``HF_HOME`` and ``HF_HUB_CACHE``).
    :return: Path to the local safetensors file.
    :raises ValueError: If ``variant`` or ``pretrained`` is not supported.
    :raises ImportError: If ``huggingface_hub`` is not installed (it ships
        with the ``nn`` extra: ``pip install "pyvisim[nn]"``).
    :raises huggingface_hub.errors.HfHubHTTPError: If the download fails.
    """
    spec = get_checkpoint_spec(variant, pretrained)
    _hf_hub_import.check()
    return Path(hf_hub_download(spec.repo_id, spec.filename, cache_dir=cache_dir))
