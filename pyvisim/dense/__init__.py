"""
Dense similarity metrics.

Metrics that compare two aligned images directly instead of going through an
intermediate vector embedding. Every metric derives from
:class:`~pyvisim.base.DenseMetricBase`, which owns input normalization, shape
validation and memory-bounded pair batching.
"""

from . import pixelwise, structural

__all__ = ["pixelwise", "structural"]
