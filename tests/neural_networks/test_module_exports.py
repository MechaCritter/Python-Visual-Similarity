"""Import-surface tests for the ``pyvisim.neural_networks`` package."""

from __future__ import annotations

import importlib
import pkgutil

import pyvisim.neural_networks


def test_public_exports() -> None:
    """The documented classes are importable from their canonical locations."""
    from pyvisim.neural_networks import (
        BCESiameseNetwork,
        ContrastiveSiameseNetwork,
        TripletNeuralNetwork,
    )
    from pyvisim.neural_networks.losses import ContrastiveLoss, TripletLoss
    from pyvisim.neural_networks.siamese import ResNetBackbone

    assert ContrastiveSiameseNetwork is not None
    assert BCESiameseNetwork is not None
    assert TripletNeuralNetwork is not None
    assert ContrastiveLoss is not None
    assert TripletLoss is not None
    assert ResNetBackbone is not None


def test_package_all_matches_exports() -> None:
    """``__all__`` of the package lists only real attributes."""
    for name in pyvisim.neural_networks.__all__:
        assert hasattr(pyvisim.neural_networks, name)


def test_every_submodule_is_importable() -> None:
    """All modules shipped under ``pyvisim.neural_networks`` import cleanly.

    A module that raises on import (e.g. a stale file left behind by a
    refactoring) would break ``import *``-style discovery, documentation
    builds and type checking.
    """
    package = pyvisim.neural_networks
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        importlib.import_module(module_info.name)
