"""
Structural similarity metrics.

Dense metrics that compare two aligned images through local luminance,
contrast and structure statistics instead of an intermediate vector encoding:
the single-scale :class:`SSIM` (Wang et al., 2004) and the multi-scale
:class:`MSSSIM` (Wang et al., 2003).
"""

from ._ms_ssim import MSSSIM
from ._ssim import SSIM

__all__ = ["SSIM", "MSSSIM"]
