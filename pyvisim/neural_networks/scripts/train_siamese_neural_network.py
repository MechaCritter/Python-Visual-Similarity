import os
import random
from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ...datasets import OxfordFlowerDataset
from ..losses import ContrastiveLoss
from ..siamese import ContrastiveSiameseNetwork

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class _RandomSource(Protocol):
    """
    Minimal random interface for pair sampling.

    Satisfied by both the :mod:`random` module (used for fresh per-epoch
    sampling in workers) and a :class:`random.Random` instance (used to
    precompute a fixed pair set).
    """

    def random(self) -> float: ...

    def choice(self, seq: Sequence[int]) -> int: ...


def _seed_worker(worker_id: int) -> None:
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

# The Resize(256) + CenterCrop(224) geometry mirrors the model's default
# inference transform (ContrastiveSiameseNetwork._get_imagenet_transform), so the
# network is trained and evaluated on the same crop distribution.
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
    Builds contrastive image pairs on top of :class:`OxfordFlowerDataset`.

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
        # random state that per-worker seeding perturbs (see _seed_worker).
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


CONFIG: dict[str, Any] = {
    "embedding_dim": 128,
    "margin": 1.0,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "num_epochs": 20,
    "num_workers": 4,
    "checkpoint_dir": "checkpoints",
    "log_every_n": 50,
}


def run_epoch(
    model: ContrastiveSiameseNetwork,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    is_train: bool,
    epoch: int,
) -> float:
    """
    Runs one training or evaluation epoch over the pair loader.

    :param model: Siamese neural network being trained or evaluated.
    :param loader: Data loader yielding ``(img_a, img_b, label)`` batches.
    :param criterion: Contrastive loss function.
    :param optimizer: Optimizer used when ``is_train`` is ``True``.
    :param is_train: ``True`` to train (grads enabled), ``False`` to evaluate.
    :param epoch: Current epoch number, used for logging.
    :return: The average loss over the epoch.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for step, (img_a, img_b, labels) in enumerate(loader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)

            emb_a = model(img_a)
            emb_b = model(img_b)
            loss = criterion(emb_a, emb_b, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            if is_train and (step + 1) % CONFIG["log_every_n"] == 0:
                print(
                    f"  Epoch {epoch:03d} | step {step + 1:04d} "
                    f"| loss {total_loss / (step + 1):.4f}"
                )

    return total_loss / len(loader)


def train() -> None:
    """Trains the Siamese network on Oxford Flowers, checkpointing the best model."""
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    train_ds = OxfordSiamesePairs("train", transform=TRAIN_TF)
    val_ds = OxfordSiamesePairs("validation", transform=VAL_TF, deterministic=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")
    print(f"Flower classes: {len(train_ds._labels_list)}")

    model = ContrastiveSiameseNetwork(
        backbone="resnet18",
        embedding_dim=CONFIG["embedding_dim"],
        device=device,
        pretrained_backbone=True,
    )

    criterion = ContrastiveLoss(margin=CONFIG["margin"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )

    best_val = float("inf")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, is_train=True, epoch=epoch
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, is_train=False, epoch=epoch
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| lr {scheduler.get_last_lr()[0]:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "cfg": CONFIG,
        }
        torch.save(ckpt, f"{CONFIG['checkpoint_dir']}/latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, f"{CONFIG['checkpoint_dir']}/best.pt")
            print(f"New best val loss: {best_val:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


def demo(checkpoint: str, img_path_a: str, img_path_b: str) -> None:
    """
    Compares two flower images using a trained checkpoint.

    :param checkpoint: Path to a checkpoint saved by :func:`train`.
    :param img_path_a: Path to the first image.
    :param img_path_b: Path to the second image.
    """
    import numpy as np
    from PIL import Image

    model = ContrastiveSiameseNetwork(
        backbone="resnet18",
        embedding_dim=CONFIG["embedding_dim"],
        device=device,
        pretrained_backbone=False,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    img_a = np.asarray(Image.open(img_path_a).convert("RGB"))
    img_b = np.asarray(Image.open(img_path_b).convert("RGB"))
    score = float(model.similarity_score(img_a, img_b).item())
    print(f"Similarity: {score:.4f}  (range -1 to 1, higher = more similar)")


if __name__ == "__main__":
    train()
