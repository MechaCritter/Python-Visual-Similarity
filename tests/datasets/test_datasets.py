"""Tests for :mod:`pyvisim.datasets.datasets`.

All network and filesystem boundaries are mocked; the suite never downloads or
reads the real Oxford-102 Flowers data. Symbols are patched where they are
*used*, i.e. as attributes of ``pyvisim.datasets.datasets``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import numpy as np
import pytest
from torchvision import transforms

from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.datasets import datasets as ds
from pyvisim.datasets.datasets import (
    NUM_TEST_IMG,
    NUM_TRAIN_IMG,
    NUM_VAL_IMG,
    OXFORD_NUM_IMAGES,
)

#: Five fake image file names returned by a mocked ``os.listdir``.
_FAKE_JPGS = [f"image_{i:05d}.jpg" for i in range(1, 6)]


def _fake_loadmat(path: str, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """A small, internally consistent fake of ``scipy.io.loadmat`` for 5 images.

    The setid arrays follow the dataset's deliberate train/test swap: ``tstid``
    becomes the train split and ``trnid`` becomes the test split.

    :param path: the ``.mat`` file path being loaded.
    :returns: the labels or setid dictionary, chosen by file name.
    """
    if str(path).endswith("labels.mat"):
        return {"labels": np.array([[1, 1, 2, 2, 3]])}
    return {
        "tstid": np.array([[1, 2]]),  # -> train (swapped)
        "valid": np.array([[3]]),  # -> validation
        "trnid": np.array([[4, 5]]),  # -> test (swapped)
    }


def _good_loadmat(path: str, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """A fake ``loadmat`` whose lengths match the real dataset (integrity check).

    :param path: the ``.mat`` file path being loaded.
    :returns: full-length labels or setid arrays.
    """
    if str(path).endswith("labels.mat"):
        return {"labels": np.ones((1, OXFORD_NUM_IMAGES), dtype=int)}
    return {
        "tstid": np.arange(NUM_TEST_IMG).reshape(1, -1),
        "valid": np.arange(NUM_VAL_IMG).reshape(1, -1),
        "trnid": np.arange(NUM_TRAIN_IMG).reshape(1, -1),
    }


@contextlib.contextmanager
def _construction_io(
    *,
    downloaded: bool = True,
    integrity: bool = True,
    loadmat: Callable[..., dict[str, np.ndarray]] = _fake_loadmat,
    listdir: list[str] | None = None,
) -> Iterator[mock.MagicMock]:
    """Patch the IO boundaries used while constructing an ``OxfordFlowerDataset``.

    :param downloaded: value returned by the mocked ``_data_downloaded``.
    :param integrity: value returned by the mocked ``_check_data_integrity``.
    :param loadmat: side effect for the mocked ``scipy.io.loadmat``.
    :param listdir: return value for the mocked ``os.listdir``.
    :yields: the mock standing in for ``download_oxford_flowers_data``.
    """
    listing = _FAKE_JPGS if listdir is None else listdir
    with (
        mock.patch.object(ds, "_data_downloaded", return_value=downloaded),
        mock.patch.object(ds, "_check_data_integrity", return_value=integrity),
        mock.patch.object(ds, "download_oxford_flowers_data") as download_mock,
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=loadmat),
        mock.patch.object(ds.os, "listdir", return_value=listing),
    ):
        yield download_mock


# §4.1 _data_downloaded


def test_data_downloaded_true() -> None:
    """Both the root dir and the label file present means the data is downloaded."""
    with (
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.os.path, "isfile", return_value=True),
    ):
        assert ds._data_downloaded() is True


def test_data_downloaded_missing_root() -> None:
    """A missing root directory means the data is not downloaded."""
    with (
        mock.patch.object(ds.os.path, "isdir", return_value=False),
        mock.patch.object(ds.os.path, "isfile", return_value=True),
    ):
        assert ds._data_downloaded() is False


def test_data_downloaded_missing_label_file() -> None:
    """A missing label file means the data is not downloaded."""
    with (
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.os.path, "isfile", return_value=False),
    ):
        assert ds._data_downloaded() is False


# §4.2 _check_data_integrity


def test_integrity_all_good() -> None:
    """Correct label count, setid lengths and image count pass the integrity check."""
    all_jpgs = [f"image_{i:05d}.jpg" for i in range(1, OXFORD_NUM_IMAGES + 1)]
    with (
        mock.patch.object(ds.os.path, "isfile", return_value=True),
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=_good_loadmat),
        mock.patch.object(ds.os, "listdir", return_value=all_jpgs),
    ):
        assert ds._check_data_integrity() is True


def test_integrity_missing_label_file() -> None:
    """A missing label file fails the integrity check."""
    with mock.patch.object(ds.os.path, "isfile", return_value=False):
        assert ds._check_data_integrity() is False


def test_integrity_wrong_label_count() -> None:
    """A wrong number of labels fails the integrity check."""

    def bad_labels(path: str, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        if str(path).endswith("labels.mat"):
            return {"labels": np.ones((1, 10), dtype=int)}
        return _good_loadmat(path)

    all_jpgs = [f"image_{i:05d}.jpg" for i in range(1, OXFORD_NUM_IMAGES + 1)]
    with (
        mock.patch.object(ds.os.path, "isfile", return_value=True),
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=bad_labels),
        mock.patch.object(ds.os, "listdir", return_value=all_jpgs),
    ):
        assert ds._check_data_integrity() is False


def test_integrity_wrong_setid_lengths() -> None:
    """Wrong setid split lengths fail the integrity check."""

    def bad_setid(path: str, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        if str(path).endswith("labels.mat"):
            return {"labels": np.ones((1, OXFORD_NUM_IMAGES), dtype=int)}
        return {
            "tstid": np.array([[1]]),
            "valid": np.array([[1, 2]]),
            "trnid": np.array([[1, 2, 3]]),
        }

    all_jpgs = [f"image_{i:05d}.jpg" for i in range(1, OXFORD_NUM_IMAGES + 1)]
    with (
        mock.patch.object(ds.os.path, "isfile", return_value=True),
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=bad_setid),
        mock.patch.object(ds.os, "listdir", return_value=all_jpgs),
    ):
        assert ds._check_data_integrity() is False


def test_integrity_wrong_image_count() -> None:
    """A wrong number of image files fails the integrity check."""
    with (
        mock.patch.object(ds.os.path, "isfile", return_value=True),
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=_good_loadmat),
        mock.patch.object(ds.os, "listdir", return_value=_FAKE_JPGS),
    ):
        assert ds._check_data_integrity() is False


def test_integrity_loadmat_raises() -> None:
    """An error while reading the ``.mat`` files is caught and returns ``False``."""
    all_jpgs = [f"image_{i:05d}.jpg" for i in range(1, OXFORD_NUM_IMAGES + 1)]
    with (
        mock.patch.object(ds.os.path, "isfile", return_value=True),
        mock.patch.object(ds.os.path, "isdir", return_value=True),
        mock.patch.object(ds.scipy.io, "loadmat", side_effect=Exception("boom")),
        mock.patch.object(ds.os, "listdir", return_value=all_jpgs),
    ):
        assert ds._check_data_integrity() is False


# ---------------------------------------------------------------------------
# §4.3 download_oxford_flowers_data
# ---------------------------------------------------------------------------


def test_download_spawns_three_processes() -> None:
    """One process is spawned per file (images/labels/setid), started and joined."""
    with (
        mock.patch.object(ds.os, "makedirs"),
        mock.patch.object(ds, "Process") as process_cls,
    ):
        ds.download_oxford_flowers_data()
    assert process_cls.call_count == 3
    assert process_cls.return_value.start.call_count == 3
    assert process_cls.return_value.join.call_count == 3


def test_download_makes_root_dir() -> None:
    """The dataset root directory is created with ``exist_ok=True``."""
    with (
        mock.patch.object(ds.os, "makedirs") as makedirs_mock,
        mock.patch.object(ds, "Process"),
    ):
        ds.download_oxford_flowers_data()
    makedirs_mock.assert_any_call(ds._DATASET_ROOT, exist_ok=True)


# §4.4 _download_and_process_file / _download_file_with_progress


def test_process_zip_dispatch() -> None:
    """A ``.zip`` destination is unzipped and then removed."""
    with (
        mock.patch.object(ds, "_download_file_with_progress"),
        mock.patch.object(ds, "_extract_zip") as extract_zip,
        mock.patch.object(ds, "_extract_tar") as extract_tar,
        mock.patch.object(ds.os, "remove") as remove,
    ):
        ds._download_and_process_file("http://x", "a.zip", "/tmp/extract")
    assert extract_zip.call_count == 1
    assert extract_tar.call_count == 0
    remove.assert_called_once_with("a.zip")


def test_process_tgz_dispatch() -> None:
    """A ``.tgz`` destination is untarred and then removed."""
    with (
        mock.patch.object(ds, "_download_file_with_progress"),
        mock.patch.object(ds, "_extract_zip") as extract_zip,
        mock.patch.object(ds, "_extract_tar") as extract_tar,
        mock.patch.object(ds.os, "remove") as remove,
    ):
        ds._download_and_process_file("http://x", "a.tgz", "/tmp/extract")
    assert extract_tar.call_count == 1
    assert extract_zip.call_count == 0
    remove.assert_called_once_with("a.tgz")


def test_process_other_no_extract() -> None:
    """A non-archive destination is neither extracted nor removed."""
    with (
        mock.patch.object(ds, "_download_file_with_progress"),
        mock.patch.object(ds, "_extract_zip") as extract_zip,
        mock.patch.object(ds, "_extract_tar") as extract_tar,
        mock.patch.object(ds.os, "remove") as remove,
    ):
        ds._download_and_process_file("http://x", "a.mat", "/tmp/extract")
    assert extract_zip.call_count == 0
    assert extract_tar.call_count == 0
    assert remove.call_count == 0


def test_download_writes_content() -> None:
    """Streamed response chunks are written to the destination file handle."""

    class _FakeResponse:
        headers = {"content-length": "8"}

        def iter_content(self, chunk_size: int = 1) -> list[bytes]:
            return [b"abcd", b"efgh"]

        def raise_for_status(self) -> None:
            return None

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    open_mock = mock.mock_open()
    with (
        mock.patch.object(ds.requests, "get", return_value=_FakeResponse()),
        mock.patch("builtins.open", open_mock),
    ):
        ds._download_file_with_progress("http://x", "/tmp/f.bin")
    handle = open_mock()
    handle.write.assert_any_call(b"abcd")
    handle.write.assert_any_call(b"efgh")


# §4.5 OxfordFlowerDataset


def test_transform_not_none_raises() -> None:
    """Supplying a transform raises ``NotImplementedError`` (not yet supported)."""
    with _construction_io():
        with pytest.raises(
            NotImplementedError, match="Transformations are not yet supported"
        ):
            OxfordFlowerDataset(transform=transforms.Compose([]))


def test_duplicate_purposes_raises() -> None:
    """Duplicate purposes raise ``ValueError``."""
    with _construction_io():
        with pytest.raises(ValueError, match="Duplicate purposes"):
            OxfordFlowerDataset(purpose=["train", "train"])


def test_unknown_purpose_raises() -> None:
    """An unknown purpose raises ``ValueError``."""
    with _construction_io():
        with pytest.raises(ValueError, match="Unknown purpose: foo"):
            OxfordFlowerDataset(purpose="foo")


def test_triggers_download_when_missing() -> None:
    """A missing/invalid dataset triggers a download exactly once."""
    with _construction_io(downloaded=False, integrity=False) as download_mock:
        OxfordFlowerDataset(purpose="train")
    assert download_mock.call_count == 1


def test_no_download_when_present() -> None:
    """A present and valid dataset does not trigger a download."""
    with _construction_io() as download_mock:
        OxfordFlowerDataset(purpose="train")
    assert download_mock.call_count == 0


def test_len_matches_filtered_train() -> None:
    """``len`` reflects the filtered train split (the swapped ``tstid`` = ``[1, 2]``)."""
    with _construction_io():
        dataset = OxfordFlowerDataset(purpose="train")
    assert len(dataset) == 2


def test_filter_by_purpose_selects_correct_paths() -> None:
    """The train split selects the images and labels for ids 1 and 2."""
    with _construction_io():
        dataset = OxfordFlowerDataset(purpose="train")
    basenames = [os.path.basename(path) for path in dataset.image_paths]
    assert basenames == ["image_00001.jpg", "image_00002.jpg"]
    assert list(dataset.labels) == [1, 1]


def test_combined_purposes_dedup() -> None:
    """Combining train and validation yields the deduplicated id set {1, 2, 3}."""
    with _construction_io():
        dataset = OxfordFlowerDataset(purpose=["train", "validation"])
    assert len(dataset) == 3


def test_getitem_returns_triple() -> None:
    """Indexing returns an ``(image, label, path)`` triple."""
    with _construction_io():
        dataset = OxfordFlowerDataset(purpose="train")
        with mock.patch.object(
            ds, "read_image_rgb", return_value=np.zeros((256, 256, 3), np.uint8)
        ):
            image, label, path = dataset[0]
    assert image.shape == (256, 256, 3)
    assert label == 1
    assert isinstance(path, str)


def test_getitem_calls_read_image_rgb() -> None:
    """Indexing reads the image at the corresponding path via ``read_image_rgb``."""
    with _construction_io():
        dataset = OxfordFlowerDataset(purpose="train")
        with mock.patch.object(
            ds, "read_image_rgb", return_value=np.zeros((256, 256, 3), np.uint8)
        ) as read_mock:
            dataset[0]
    assert read_mock.call_args[0][0] == dataset.image_paths[0]
