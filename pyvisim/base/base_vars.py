#: Dynamic range of the canonical pixel representation. Inputs are rescaled
#: into the canonical ``uint8`` ``[0, 255]`` range before scoring, so metrics
#: that need a data range (SSIM, PSNR, ...) always use this value.
CANONICAL_DATA_RANGE = 255.0
