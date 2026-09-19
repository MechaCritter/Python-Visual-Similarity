"""
Dense similarity metrics.

Metrics that compare two aligned images directly instead of going through an
intermediate vector embedding.
"""

from . import pixelwise, structural

__all__ = ["pixelwise", "structural"]
