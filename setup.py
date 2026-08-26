"""Build script for the compiled extension modules."""

import os
import sys
import tempfile

import numpy
import pybind11
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

#: Root of the vendored hnswlib sources.
_HNSWLIB_ROOT = "pyvisim/image_store/_index/_vendored/hnswlib"

#: Directory of the sources extending the vendored search structures.
_HNSWLIB_BINDINGS_ROOT = "pyvisim/image_store/_index/_bindings"

#: Name of the C++ extension built from the hnswlib bindings. Only this
#: extension receives the C++ compile flags probed by :class:`BuildExt`.
_HNSWLIB_EXT_NAME = "pyvisim.image_store._index._bindings._hnswlib"

#: ``-march=native`` tunes the binary for the building machine, which is wrong
#: for a redistributable wheel. Set this environment variable to drop it.
_NO_NATIVE_ENV_VAR = "PYVISIM_NO_NATIVE"

#: Tuning flag enabling the CPU-specific instructions of the build machine.
_NATIVE_FLAG = "-march=native"


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


def _has_flag(compiler: object, flag: str) -> bool:
    """
    Report whether the active compiler accepts a command-line flag.

    :param compiler: The ``distutils`` compiler instance of the build.
    :param flag: The flag to probe, e.g. ``"-march=native"``.
    :return: ``True`` if a trivial translation unit compiles with ``flag``.
    """
    with tempfile.TemporaryDirectory() as probe_dir:
        source_path = os.path.join(probe_dir, "probe.cpp")
        with open(source_path, "w") as source:
            source.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile(  # type: ignore[attr-defined]
                [source_path], output_dir=probe_dir, extra_postargs=[flag]
            )
        except Exception:
            return False
    return True


def _cpp_standard_flag(compiler: object) -> str:
    """
    Return the C++ standard flag to build the hnswlib bindings with.

    :param compiler: The ``distutils`` compiler instance of the build.
    :return: ``-std=c++14`` when available, ``-std=c++11`` otherwise.
    :raises RuntimeError: If the compiler supports neither standard.
    """
    for flag in ("-std=c++14", "-std=c++11"):
        if _has_flag(compiler, flag):
            return flag
    raise RuntimeError("Unsupported compiler: at least C++11 support is needed.")


class BuildExt(build_ext):
    """
    Build command adding the compiler-specific flags of the C++ extension.

    The flags are the ones the hnswlib authors build their bindings with. They
    are probed against the active compiler and applied to
    :data:`_HNSWLIB_EXT_NAME` only, leaving the Cython extensions with the flags
    declared on them.
    """

    #: Base compile flags per compiler family.
    compile_options = {
        "msvc": ["/EHsc", "/openmp", "/O2"],
        "unix": ["-O3", _NATIVE_FLAG],
    }
    #: Base link flags per compiler family.
    link_options: dict[str, list[str]] = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        compile_options["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        link_options["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    else:
        compile_options["unix"] += ["-fopenmp"]
        link_options["unix"] += ["-fopenmp", "-pthread"]

    def build_extensions(self) -> None:
        """
        Extend the C++ extension with the probed flags, then build everything.

        :return: ``None``.
        """
        family = self.compiler.compiler_type
        options = list(self.compile_options.get(family, []))
        if family == "unix":
            options = self._unix_options(options)

        for extension in self.extensions:
            if extension.name != _HNSWLIB_EXT_NAME:
                continue
            extension.extra_compile_args.extend(options)
            extension.extra_link_args.extend(self.link_options.get(family, []))
        build_ext.build_extensions(self)

    def _unix_options(self, options: list[str]) -> list[str]:
        """
        Resolve the flags that depend on what the Unix compiler supports.

        :param options: The base Unix compile flags.
        :return: The flags to compile the C++ extension with.
        """
        options.append(_cpp_standard_flag(self.compiler))
        if _has_flag(self.compiler, "-fvisibility=hidden"):
            options.append("-fvisibility=hidden")
        if os.environ.get(_NO_NATIVE_ENV_VAR) or not _has_flag(
            self.compiler, _NATIVE_FLAG
        ):
            options.remove(_NATIVE_FLAG)
        return options


def _extensions() -> list[Extension]:
    """
    Declare the compiled extension modules of the package.

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
        Extension(
            "pyvisim.pixelwise._kernel._ssd_kernel",
            ["pyvisim/pixelwise/_kernel/_ssd_kernel.pyx"],
            extra_compile_args=optimize_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
        ),
    ]


def _hnswlib_extension() -> Extension:
    """
    Declare the C++ extension built on the vendored hnswlib bindings.

    The vendored sources are pulled in by the bindings of this package, which
    extend them, so the extension is built from a single translation unit.

    :return: The pybind11 ``Extension`` exposing the ``Index``/``BFIndex``
        search structures.
    """
    return Extension(
        _HNSWLIB_EXT_NAME,
        [f"{_HNSWLIB_BINDINGS_ROOT}/_hnswlib.cpp"],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            f"{_HNSWLIB_ROOT}/hnswlib",
        ],
        depends=[
            f"{_HNSWLIB_BINDINGS_ROOT}/pyvisim_index.h",
            f"{_HNSWLIB_BINDINGS_ROOT}/upstream_index.h",
            f"{_HNSWLIB_ROOT}/python-bindings/bindings.cpp",
        ],
        language="c++",
    )


setup(
    ext_modules=cythonize(_extensions(), language_level="3") + [_hnswlib_extension()],
    cmdclass={"build_ext": BuildExt},
)
