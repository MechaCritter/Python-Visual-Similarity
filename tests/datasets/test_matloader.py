"""Tests for :mod:`pyvisim.datasets._matloader` to see if its behavior is
consistent with SciPy's ``scipy.io.loadmat`` on real Oxford-102 Flowers ``.mat`` files.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyvisim.datasets._matloader import load_mat
from pyvisim.datasets.datasets import (
    _DATASET_ROOT,
    _IMAGE_LABEL_FILE,
    _SETID_FILE,
    download_label,
    download_setid,
)

scipy_io = pytest.importorskip("scipy.io")


@pytest.fixture(scope="session")
def oxford_mat_files() -> dict[str, str]:
    """Download the real Oxford-102 ``.mat`` files via the dataset downloaders.

    The label and set-id files are fetched into the pyvisim cache directory
    with the library's own :func:`~pyvisim.datasets.datasets.download_label`
    and :func:`~pyvisim.datasets.datasets.download_setid` helpers, mirroring
    how the dataset is obtained in production. They are never committed.

    :return: Mapping of ``.mat`` file name to its downloaded path.
    """
    os.makedirs(_DATASET_ROOT, exist_ok=True)
    download_label()
    download_setid()
    return {"labels.mat": _IMAGE_LABEL_FILE, "setid.mat": _SETID_FILE}


def _reference(path: str) -> dict[str, np.ndarray]:
    """Load a ``.mat`` file with SciPy, dropping its metadata keys.

    :param path: Path to the ``.mat`` file.
    :return: Mapping of variable name to array, without the ``__*__`` entries.
    """
    loaded = scipy_io.loadmat(path)
    return {key: value for key, value in loaded.items() if not key.startswith("__")}


@pytest.mark.parametrize("filename", ["labels.mat", "setid.mat"])
def test_load_mat_matches_scipy(
    filename: str, oxford_mat_files: dict[str, str]
) -> None:
    """``load_mat`` matches SciPy on the real Oxford flower ``.mat`` files.

    The variable set, and every array's dtype, shape and values must be
    identical to what ``scipy.io.loadmat`` returns.
    """
    path = oxford_mat_files[filename]

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
