"""Shared pair-sampling data pipeline for the Siamese training scripts.

Both training scripts (contrastive and pairwise) learn from ``(img_a, img_b,
label)`` triples where ``label = 1`` marks a same-class pair and ``label = 0``
a cross-class one, so the pair mining, the image transforms and the DataLoader
worker seeding live here.
"""

import random
from collections.abc import Sequence
from typing import Protocol

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ...datasets import OxfordFlowerDataset

SEED = 42


class _RandomSource(Protocol):
    """
    Minimal random interface for pair sampling.

    Satisfied by both the :mod:`random` module (used for fresh per-epoch
    sampling in workers) and a :class:`random.Random` instance (used to
    precompute a fixed pair set).
    """

    def random(self) -> float: ...

    def choice(self, seq: Sequence[int]) -> int: ...


def seed_worker(worker_id: int) -> None:
    """
    Reseeds the :mod:`random` module inside a DataLoader worker process.

    DataLoader workers are forked from the main process, so they inherit its
    :mod:`random` state and would otherwise draw identical pair samples across
    workers (and re-draw the same stream every epoch once respawned). PyTorch
    hands each worker a distinct ``torch`` seed per epoch; this propagates it
    into the :mod:`random` module so the coin flips and partner choices are
    fresh per worker and per epoch.

    :param worker_id: Index of the worker process (required by the
        ``worker_init_fn`` interface; unused here).
    """
    random.seed(torch.initial_seed() % 2**32)


# Dataset returns uint8 HWC numpy arrays, so we convert via PIL.
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# The Resize(256) + CenterCrop(224) geometry mirrors the models' default
# inference transform (SiameseNetworkBase._get_imagenet_transform), so the
# networks are trained and evaluated on the same crop distribution.
TRAIN_TF = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)

VAL_TF = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


class OxfordSiamesePairs(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Builds labelled image pairs on top of :class:`OxfordFlowerDataset`.

    Each item is a ``(img_a, img_b, label)`` triple where ``label = 1`` marks a
    positive pair (same flower class) and ``label = 0`` a negative pair
    (different classes). Transforms are applied here since
    :class:`OxfordFlowerDataset` does not support them internally yet.

    :param purpose: Dataset split to draw from (e.g. ``"train"``).
    :param transform: Transform applied to each image.
    :param pos_fraction: Fraction of sampled pairs that are positive.
    :param deterministic: If ``True``, the full pair set is sampled once at
        construction time with a fixed seed and reused for every access, so the
        dataset yields the same pairs on every epoch. Use this for validation,
        where a fresh random pair set each epoch would make the validation loss
        (and thus best-checkpoint selection) noisy. If ``False`` (default),
        partners are resampled on every access for fresh per-epoch training
        pairs.
    :raises ValueError: If the split contains fewer than two classes.
    """

    def __init__(
        self,
        purpose: str,
        transform: transforms.Compose,
        pos_fraction: float = 0.5,
        deterministic: bool = False,
    ) -> None:
        self._base = OxfordFlowerDataset(transform=None, purpose=purpose)
        self.transform = transform
        self.pos_fraction = pos_fraction

        # Build a label -> list[index] map for fast pair mining. Labels are
        # read directly; indexing the base dataset would decode every image.
        self._label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(self._base.labels):
            self._label_to_indices.setdefault(label, []).append(idx)

        self._labels_list = list(self._label_to_indices.keys())
        if len(self._labels_list) < 2:
            raise ValueError("Need >= 2 classes for contrastive training.")

        # A fixed pair set decouples the validation loss from sampling noise.
        # A local RNG keeps it reproducible and independent of the global
        # random state that per-worker seeding perturbs (see seed_worker).
        self._pairs: list[tuple[int, int]] | None = None
        if deterministic:
            rng = random.Random(SEED)
            self._pairs = [
                self._sample_partner(idx, rng) for idx in range(len(self._base))
            ]

    def __len__(self) -> int:
        return len(self._base)

    def _get_tensor(self, idx: int) -> torch.Tensor:
        image, _, _ = self._base[idx]
        tensor: torch.Tensor = self.transform(image)
        return tensor

    def _sample_partner(self, idx: int, rng: _RandomSource) -> tuple[int, int]:
        """
        Samples a partner index and pair label for the image at ``idx``.

        :param idx: Index of the anchor image in the base dataset.
        :param rng: Random source for the positive/negative coin flip and the
            partner choice.
        :return: A ``(partner_index, pair_label)`` tuple, where ``pair_label``
            is ``1`` for a positive (same-class) pair and ``0`` for a negative
            (cross-class) one.
        """
        label_a = self._base.labels[idx]
        if rng.random() < self.pos_fraction:
            # Positive: another image of the same class.
            same_indices = [i for i in self._label_to_indices[label_a] if i != idx]
            idx_b = rng.choice(same_indices) if same_indices else idx
            return idx_b, 1
        # Negative: an image from a different class.
        neg_label = rng.choice([lbl for lbl in self._labels_list if lbl != label_a])
        idx_b = rng.choice(self._label_to_indices[neg_label])
        return idx_b, 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_a = self._get_tensor(idx)
        if self._pairs is not None:
            idx_b, pair_label = self._pairs[idx]
        else:
            idx_b, pair_label = self._sample_partner(idx, random)
        img_b = self._get_tensor(idx_b)
        return img_a, img_b, torch.tensor(pair_label, dtype=torch.float32)
