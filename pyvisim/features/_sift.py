
from .._base_classes import FeatureExtractorBase
from ._vendored.sift.sift import SIFT as _SIFT

__all__ = ["SIFT"]


class SIFT(FeatureExtractorBase, _SIFT):
    ...
