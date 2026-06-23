from typing import Any

from .._base_classes import FeatureExtractorBase
from ._deep_conv_feature import DeepConvFeature
from ._root_sift import RootSIFT
from ._sift import SIFT

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
