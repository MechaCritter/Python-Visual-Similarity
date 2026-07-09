"""Tests for :mod:`pyvisim.datasets._matloader` to see if its behavior is
consistent with SciPy's ``scipy.io.loadmat`` on real Oxford-102 Flowers ``.mat`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvisim.datasets._matloader import load_mat

scipy_io = pytest.importorskip("scipy.io")

#: Vendored real Oxford-102 Flowers ``.mat`` files, byte-for-byte copies of the
#: originals served at https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.
_FIXTURES = Path(__file__).parent / "fixtures"
_MAT_FILES = ["labels.mat", "setid.mat"]


def _reference(path: str) -> dict[str, np.ndarray]:
    """Load a ``.mat`` file with SciPy, dropping its metadata keys.

    :param path: Path to the ``.mat`` file.
    :return: Mapping of variable name to array, without the ``__*__`` entries.
    """
    loaded = scipy_io.loadmat(path)
    return {key: value for key, value in loaded.items() if not key.startswith("__")}


@pytest.mark.parametrize("filename", _MAT_FILES)
def test_load_mat_matches_scipy(filename: str) -> None:
    """``load_mat`` matches SciPy on the real Oxford flower ``.mat`` files.

    The variable set, and every array's dtype, shape and values must be
    identical to what ``scipy.io.loadmat`` returns.
    """
    path = str(_FIXTURES / filename)

    produced = load_mat(path)
    expected = _reference(path)

    assert set(produced) == set(expected)
    for key, ref_array in expected.items():
        assert produced[key].dtype == ref_array.dtype
        assert produced[key].shape == ref_array.shape
        assert np.array_equal(produced[key], ref_array)


def test_load_mat_rejects_non_matfile(tmp_path: Path) -> None:
    """A file that is not a Level-5 MAT-file raises :class:`ValueError`."""
    path = tmp_path / "not_a.mat"
    path.write_bytes(b"not a mat file")
    with pytest.raises(ValueError):
        load_mat(str(path))
