"""Fixtures shared by the image store tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest


@pytest.fixture
def run_python() -> Callable[[str], None]:
    """Run Python code in a fresh interpreter, where nothing is imported yet.

    Whether a module loads FAISS, and what happens without it, can only be seen
    before the first import of FAISS, which this test process has long done.

    :returns: a function running the given source, which fails the test with
        the interpreter's stderr if the source raises.
    """

    def run(code: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    return run
