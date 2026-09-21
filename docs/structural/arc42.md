# arc42: structural

Software architecture of `pyvisim.structural`. This document is for developers
and is not part of the published documentation.

## Building block view

`SSIM` and `MSSSIM` are dense metrics: they compare two images directly instead
of going through an intermediate vector embedding. Both derive from
`DenseMetricBase`, which owns the shared pipeline around the metric itself
(input normalization, shape validation and memory-bounded pair batching).
`MSSSIM` is `SSIM` applied to a downsampling pyramid, so the same kernel
computes the window statistics at every scale.

The window statistics are computed by a compiled kernel under
`pyvisim/structural/_kernel/`, which `make build-ext` regenerates from its
Cython source.

## Architecture decisions

### The kernel is multithreaded and its team size is an environment variable

`PYVISIM_NUM_THREADS` sets the OpenMP team size of the compiled kernels (4 by
default). It is read on every kernel call instead of being cached once, so it
can be changed at runtime through `os.environ`. `batch_size` bounds how many
image pairs enter one kernel call, which caps the peak memory for a large
gallery.

### The kernel computes in float32

The window statistics are computed in `float32` instead of `float64`. The
scores match a `float64` computation to roughly 1e-5, which the benchmark
against scikit-image and torchmetrics confirms. Since the metric is a
perceptual score, that error is far below the resolution anybody reads it at.
