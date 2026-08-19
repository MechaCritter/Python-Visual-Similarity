"""Oxford Flowers fixtures shared by the Siamese integration tests."""

from __future__ import annotations

import pytest

from pyvisim.datasets import OxfordFlowerDataset

#: Number of images taken from every flower class.
IMAGES_PER_CLASS = 20


@pytest.fixture(scope="session")
def flower_subset() -> OxfordFlowerDataset:
    """The training split reduced to the first 20 images of every class.

    :return: An ``OxfordFlowerDataset`` whose paths/labels are the subset.
    """
    dataset = OxfordFlowerDataset(purpose="train")
    selected: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.labels):
        indices = selected.setdefault(label, [])
        if len(indices) < IMAGES_PER_CLASS:
            indices.append(index)
    keep = sorted(index for indices in selected.values() for index in indices)
    dataset.image_paths = [dataset.image_paths[index] for index in keep]
    dataset.labels = [dataset.labels[index] for index in keep]
    return dataset
