from collections.abc import Callable

from ...lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

_torch_import.check()


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone with its final fully-connected layer removed.

    The classification head is stripped so the network outputs raw features;
    ``output_dim`` reports their dimensionality (512 for ResNet-18).

    :param pretrained: Whether to load the default ImageNet-pretrained weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.output_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts flattened backbone features.

        :param x: Input image tensor of shape (batch, channels, H, W).
        :return: Feature tensor of shape (batch, output_dim).
        """
        x = self.features(x)
        return x.flatten(1)


def _imagenet_transform() -> "transforms.Compose":
    """
    Builds the standard ImageNet evaluation preprocessing.

    Resizes the shortest side to 256, center-crops to 224x224, converts to a
    ``[0, 1]`` tensor and normalizes with the ImageNet channel statistics, as
    torchvision documents for the models trained on ImageNet. Reference:
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

    :return: The composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # this also rescales the pixel values to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


#: Preprocessing transform of every supported backbone, keyed by backbone name.
#: The values are builders rather than instances, so each model owns the
#: transform it is given. Registering a backbone that was not trained on
#: ImageNet is a matter of adding its own builder here.
_TRANSFORM_REGISTRY: dict[str, Callable[[], "transforms.Compose"]] = {
    "resnet18": _imagenet_transform,
}


def get_transform(backbone: str) -> "transforms.Compose":
    """
    Returns the preprocessing transform the given backbone was trained with.

    :param backbone: Name of the backbone network, e.g. ``"resnet18"``.
    :return: A fresh torchvision transform for that backbone.
    :raises ValueError: If no transform is registered for ``backbone``.
    """
    if backbone not in _TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}; no preprocessing transform is "
            f"registered for it. Supported backbones: "
            f"{', '.join(repr(name) for name in _TRANSFORM_REGISTRY)}."
        )
    return _TRANSFORM_REGISTRY[backbone]()
