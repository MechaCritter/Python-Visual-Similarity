"""Trains a :class:`PairwiseSiameseNetwork` on Oxford Flowers.

Parallel to :mod:`.train_siamese_neural_network`, but for the pair-classifying
variant (Koch et al., 2015) alone: the network scores each ``(img_a, img_b)``
pair with a same-class logit and learns with binary cross-entropy instead of a
contrastive loss on embeddings. The paper's L2 regularization term is realised
by the optimizer's ``weight_decay``.
"""

import os
import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..siamese import PairwiseSiameseNetwork
from ._pairs import SEED, TRAIN_TF, VAL_TF, OxfordSiamesePairs, seed_worker

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG: dict[str, Any] = {
    "embedding_dim": 128,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "num_epochs": 20,
    "num_workers": 4,
    "checkpoint_dir": "checkpoints_pairwise",
    "log_every_n": 50,
}


def run_epoch(
    model: PairwiseSiameseNetwork,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: torch.nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    is_train: bool,
    epoch: int,
) -> float:
    """
    Runs one training or evaluation epoch over the pair loader.

    :param model: Pairwise Siamese network being trained or evaluated.
    :param loader: Data loader yielding ``(img_a, img_b, label)`` batches.
    :param criterion: Binary cross-entropy loss on the pair logits.
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

            logits = model(img_a, img_b)
            loss = criterion(logits, labels)

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
    """Trains the pairwise network on Oxford Flowers, checkpointing the best model."""
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    train_ds = OxfordSiamesePairs("train", transform=TRAIN_TF)
    val_ds = OxfordSiamesePairs("validation", transform=VAL_TF, deterministic=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
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

    model = PairwiseSiameseNetwork(
        backbone="resnet18",
        embedding_dim=CONFIG["embedding_dim"],
        device=device,
        pretrained_backbone=True,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
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
    Classifies whether two flower images show the same class.

    :param checkpoint: Path to a checkpoint saved by :func:`train`.
    :param img_path_a: Path to the first image.
    :param img_path_b: Path to the second image.
    """
    import numpy as np
    from PIL import Image

    model = PairwiseSiameseNetwork(
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
    probability = float(model.similarity_score(img_a, img_b).item())
    print(
        f"Same-class probability: {probability:.4f}  "
        f"(range 0 to 1, higher = more likely the same class)"
    )


if __name__ == "__main__":
    train()
