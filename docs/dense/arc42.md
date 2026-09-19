# arc42: dense

Software architecture of `pyvisim.dense`. This document is for developers and
is not part of the published documentation.

## Building block view

Every metric under `pyvisim.dense` compares two images directly instead of
going through an intermediate vector embedding. They all derive from
`DenseMetricBase`, which owns the shared pipeline around the metric itself,
namely input normalization, shape validation and memory-bounded pair batching.

- [Structural](structural/arc42.md): SSIM and MSSSIM.

## Architecture decisions

### The kernels are multithreaded and their team size is an environment variable

`PYVISIM_NUM_THREADS` sets the OpenMP team size of the compiled kernels, 4 by
default, and it is read on every kernel call rather than cached, so it can be
changed at runtime through `os.environ`. `batch_size` bounds how many image
pairs enter one kernel call, which is what caps the peak memory of a large
gallery.
