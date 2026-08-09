import warnings
from typing import Any

from ._base_clustering import ClusteringModelBase
from .gmm import DiagCovarGaussianMixture
from .kmeans import KMeans
from .pca import PCA

__all__ = [
    "ClusteringModelBase",
    "KMeans",
    "DiagCovarGaussianMixture",
    "PCA",
]


def __getattr__(name: str) -> Any:
    if name == "GaussianMixtureModel":
        warnings.warn(
            "GaussianMixtureModel has been renamed to DiagCovarGaussianMixture; "
            "the old name will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        return DiagCovarGaussianMixture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
