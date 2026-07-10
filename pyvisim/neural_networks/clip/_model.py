"""PyTorch implementation of the CLIP image towers.

The two visual architectures of the CLIP model family are implemented here:
the Vision Transformer (``ViT-*`` variants) and the anti-aliased
attention-pooled ResNet (``RN*`` variants). Both are
re-implementations of the reference code in OpenAI's CLIP
(https://github.com/openai/CLIP, MIT license, Copyright (c) 2021 OpenAI) and
`open_clip <https://github.com/mlfoundations/open_clip>`_.

Only the image tower is implemented. :class:`ClipEmbedder` never runs the
text tower, so its weights are skipped entirely when a checkpoint is read.
"""

from collections import OrderedDict
from pathlib import Path
from typing import cast

from safetensors import safe_open

from ...lazy_import import OptionalImport
from ._registry import VisionConfig

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torch import nn
    from torch.nn import functional as F

_torch_import.check()

#: Prefix of the image-tower entries in open_clip-format state dicts.
_VISUAL_PREFIX = "visual."


class QuickGELU(nn.Module):
    """Sigmoid-based GELU approximation used by the original OpenAI CLIP."""

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Apply the QuickGELU activation.

        :param x: Input tensor.
        :return: ``x * sigmoid(1.702 * x)``.
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    One pre-norm transformer block: self-attention plus a two-layer MLP.

    :param width: Token embedding width of the transformer.
    :param heads: Number of attention heads.
    :param mlp_width: Hidden width of the MLP.
    :param act_layer: Activation module class used inside the MLP.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_width: int,
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, width)),
                ]
            )
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Run the block on a batch-first token sequence.

        :param x: Tokens of shape ``(N, L, width)``.
        :return: Transformed tokens of the same shape.
        """
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + cast("torch.Tensor", self.mlp(self.ln_2(x)))


class Transformer(nn.Module):
    """
    A stack of :class:`ResidualAttentionBlock` modules.

    :param width: Token embedding width.
    :param layers: Number of blocks.
    :param heads: Number of attention heads per block.
    :param mlp_width: Hidden width of each block's MLP.
    :param act_layer: Activation module class used inside the MLPs.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_width: int,
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads, mlp_width, act_layer)
            for _ in range(layers)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Run all blocks in sequence.

        :param x: Tokens of shape ``(N, L, width)``.
        :return: Transformed tokens of the same shape.
        """
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    """
    CLIP's ViT image tower: patch embedding, transformer and projection.

    :param image_size: Side length in pixels of the square model input.
    :param patch_size: Side length in pixels of one square patch.
    :param width: Token embedding width of the transformer.
    :param layers: Number of transformer blocks.
    :param heads: Number of attention heads per block.
    :param mlp_width: Hidden width of each block's MLP.
    :param embed_dim: Dimensionality of the output embedding space.
    :param act_layer: Activation module class used inside the MLPs.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_width: int,
        embed_dim: int,
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        grid = image_size // patch_size
        scale = width**-0.5
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid * grid + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_width, act_layer)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Embed a batch of preprocessed images.

        :param x: Images of shape ``(N, 3, image_size, image_size)``.
        :return: Image embeddings of shape ``(N, embed_dim)``.
        """
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0])
        return x @ self.proj


class Bottleneck(nn.Module):
    """
    Anti-aliased bottleneck block of CLIP's modified ResNet.

    All convolutions have stride 1; when ``stride > 1`` an average pool
    performs the downsampling after the second convolution instead, and the
    shortcut is average-pooled before its 1x1 convolution.

    :param inplanes: Number of input channels.
    :param planes: Number of bottleneck channels; the block outputs
        ``planes * 4`` channels.
    :param stride: Spatial downsampling factor of the block.
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample: nn.Module | None = None
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Run the block.

        :param x: Feature map of shape ``(N, inplanes, H, W)``.
        :return: Feature map of shape ``(N, planes * 4, H', W')``.
        """
        identity = self.downsample(x) if self.downsample is not None else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)


class AttentionPool2d(nn.Module):
    """
    QKV attention pooling that maps a feature map to a single embedding.

    A mean-pooled token queries the feature-map tokens through one round of
    multi-head attention, replacing the global average pool of a standard
    ResNet.

    :param spacial_dim: Side length of the square input feature map.
    :param embed_dim: Number of channels of the input feature map.
    :param num_heads: Number of attention heads.
    :param output_dim: Dimensionality of the output embedding space.
    """

    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Pool a feature map into one embedding per image.

        :param x: Feature map of shape ``(N, embed_dim, H, W)``.
        :return: Embeddings of shape ``(N, output_dim)``.
        """
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        pooled, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return pooled.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    CLIP's modified ResNet image tower.

    Differs from a torchvision ResNet in three ways: a 3-convolution stem
    with an average pool instead of a max pool, anti-aliased downsampling
    (average pool before every strided shortcut), and attention pooling
    instead of global average pooling.

    :param layers: Number of bottleneck blocks in each of the four stages.
    :param width: Number of channels after the stem.
    :param heads: Number of attention-pooling heads.
    :param image_size: Side length in pixels of the square model input.
    :param embed_dim: Dimensionality of the output embedding space.
    """

    def __init__(
        self,
        *,
        layers: tuple[int, int, int, int],
        width: int,
        heads: int,
        image_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width // 2, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.attnpool = AttentionPool2d(image_size // 32, width * 32, heads, embed_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Build one ResNet stage and advance the running channel count.

        :param planes: Bottleneck channels of the stage.
        :param blocks: Number of bottleneck blocks in the stage.
        :param stride: Spatial downsampling factor of the first block.
        :return: The stage as a sequential module.
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self._inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _stem(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Run the 3-convolution stem.

        :param x: Images of shape ``(N, 3, image_size, image_size)``.
        :return: Feature map downsampled by a factor of 4.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return cast("torch.Tensor", self.avgpool(x))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Embed a batch of preprocessed images.

        :param x: Images of shape ``(N, 3, image_size, image_size)``.
        :return: Image embeddings of shape ``(N, embed_dim)``.
        """
        x = self._stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return cast("torch.Tensor", self.attnpool(x))


def build_vision_model(config: VisionConfig, *, quick_gelu: bool) -> nn.Module:
    """
    Build the image tower described by a variant's vision configuration.

    :param config: Architecture description of the image tower.
    :param quick_gelu: Whether the transformer MLPs use :class:`QuickGELU`
        (original OpenAI activation) instead of :class:`torch.nn.GELU`.
        Ignored by the ResNet variants, which only use ReLU.
    :return: The randomly initialized image tower.
    :raises ValueError: If a ViT configuration carries no ``patch_size``.
    """
    if isinstance(config.layers, tuple):
        return ModifiedResNet(
            layers=config.layers,
            width=config.width,
            heads=config.heads,
            image_size=config.image_size,
            embed_dim=config.embed_dim,
        )
    if config.patch_size is None:
        raise ValueError("A Vision Transformer configuration requires a patch_size.")
    act_layer: type[nn.Module] = QuickGELU if quick_gelu else nn.GELU
    return VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        mlp_width=config.mlp_width,
        embed_dim=config.embed_dim,
        act_layer=act_layer,
    )


def load_vision_weights(model: nn.Module, path: str | Path) -> None:
    """
    Load the image-tower weights of an open_clip-format safetensors file.

    Only the ``visual.*`` entries of the checkpoint are read (the text tower
    is skipped entirely) and every tensor is cast to ``float32``, so the
    model produces identical embeddings on every device. The loaded state
    must cover the model exactly; BatchNorm ``num_batches_tracked`` counters
    are the only entries allowed to be absent, as they do not affect
    inference.

    :param model: Image tower built by :func:`build_vision_model`.
    :param path: Path of the safetensors checkpoint.
    :raises ValueError: If the checkpoint holds no image tower, or its
        entries do not match the model's parameters.
    """
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith(_VISUAL_PREFIX):
                tensor = checkpoint.get_tensor(key)
                state[key[len(_VISUAL_PREFIX) :]] = tensor.to(torch.float32)
    if not state:
        raise ValueError(
            f"{path} holds no image-tower weights: no entry starts with "
            f"'{_VISUAL_PREFIX}'. Expected an open_clip-format checkpoint."
        )
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise ValueError(
            f"{path} does not match the model architecture: {error}"
        ) from None
    missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(
            f"{path} does not match the model architecture. "
            f"Missing entries: {missing or 'none'}. "
            f"Unexpected entries: {list(unexpected) or 'none'}."
        )
