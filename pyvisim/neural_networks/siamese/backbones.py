from ...lazy_import import OptionalImport

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    import torch.nn as nn
    from torchvision import models

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
