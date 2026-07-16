#: Environment variable overriding the OpenMP team size of the compiled kernels.
_KERNEL_THREADS_ENV_VAR = "PYVISIM_NUM_THREADS"

#: OpenMP team size used by the compiled kernels when the variable is unset.
_DEFAULT_KERNEL_THREADS = 4


def get_kernel_threads() -> int:
    """
    Resolves the OpenMP team size used by the compiled kernels.

    The value is read from the ``PYVISIM_NUM_THREADS`` environment variable on
    every kernel call, so it can also be changed at runtime through
    ``os.environ``. When the variable is unset, a fixed default of
    :data:`_DEFAULT_KERNEL_THREADS` threads is used.

    :return: The number of OpenMP threads (at least 1).
    :raises ValueError: If the environment variable is set to anything other
        than a positive integer.
    """
    import os

    raw_value = os.environ.get(_KERNEL_THREADS_ENV_VAR)
    if raw_value is None:
        return _DEFAULT_KERNEL_THREADS
    try:
        num_threads = int(raw_value)
    except ValueError:
        raise ValueError(
            f"{_KERNEL_THREADS_ENV_VAR} must be a positive integer, got {raw_value!r}."
        ) from None
    if num_threads < 1:
        raise ValueError(
            f"{_KERNEL_THREADS_ENV_VAR} must be a positive integer, got {raw_value!r}."
        )
    return num_threads
