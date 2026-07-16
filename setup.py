"""Build script for the compiled extension modules."""

import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def _openmp_args() -> tuple[list[str], list[str]]:
    """
    Return the per-platform ``(compile_args, link_args)`` enabling OpenMP.

    Apple clang ships without OpenMP support, so macOS builds fall back to a
    serial kernel: ``cython.parallel.prange`` degrades to an ordinary loop.

    :return: The extra compile and link arguments for the current platform.
    """
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        return [], []
    return ["-fopenmp"], ["-fopenmp"]


def _extensions() -> list[Extension]:
    """
    Declare the Cython extension modules of the package.

    :return: One ``Extension`` per compiled module.
    """
    openmp_compile_args, openmp_link_args = _openmp_args()
    optimize_args = [] if sys.platform == "win32" else ["-O3"]
    return [
        Extension(
            "pyvisim.structural._kernel._ssim_kernels",
            ["pyvisim/structural/_kernel/_ssim_kernels.pyx"],
            extra_compile_args=optimize_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
        ),
        Extension(
            "pyvisim.features._vendored.sift._sift",
            ["pyvisim/features/_vendored/sift/_sift.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=optimize_args,
        ),
    ]


setup(ext_modules=cythonize(_extensions(), language_level="3"))
