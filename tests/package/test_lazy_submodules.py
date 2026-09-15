"""Tests for the lazily imported submodules of :mod:`pyvisim`."""

from __future__ import annotations

import subprocess
import sys

import pytest

import pyvisim

#: Every submodule reachable as an attribute, the ``nn`` ones included.
SUBMODULES = [*pyvisim.__all__, "datasets", "neural_networks"]


def _run_python(code: str) -> None:
    """Run ``code`` in a fresh interpreter, so no submodule is imported yet.

    :param code: the Python source to run.
    """
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_submodule_is_imported_on_first_access() -> None:
    """``pyvisim.<name>`` works after a plain ``import pyvisim``."""
    checks = "".join(
        f"assert pyvisim.{name}.__name__ == 'pyvisim.{name}'\n" for name in SUBMODULES
    )
    _run_python("import pyvisim\n" + checks)


def test_star_import_needs_no_torch() -> None:
    """``from pyvisim import *`` works on an install without the ``nn`` extra."""
    _run_python(
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['torchvision'] = None\n"
        "from pyvisim import *\n"
    )


def test_an_unknown_attribute_raises() -> None:
    """A name that is no submodule still raises ``AttributeError``."""
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        _ = pyvisim.missing
