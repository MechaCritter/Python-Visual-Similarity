import os
import random
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ..datasets import OxfordFlowerDataset
from .losses import ContrastiveLoss
from .siamese_neural_network import SiameseNeuralNetwork

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone with its final fully-connected layer removed.

    The classification head is stripped so the network outputs raw features;
    ``output_dim`` reports their dimensionality (512 for ResNet-18).

    :param pretrained: Whether to load the default ImageNet-pretrained weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.output_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts flattened backbone features.

        :param x: Input image tensor of shape (batch, channels, H, W).
        :return: Feature tensor of shape (batch, output_dim).
        """
        x = self.features(x)
        return x.flatten(1)


# Dataset returns uint8 HWC numpy arrays, so we convert via PIL.
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRAIN_TF = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)

VAL_TF = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
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
    """

    def __init__(
        self, purpose: str, transform: transforms.Compose, pos_fraction: float = 0.5
    ) -> None:
        self._base = OxfordFlowerDataset(transform=None, purpose=purpose)
        self.transform = transform
        self.pos_fraction = pos_fraction

        # Build a label -> list[index] map for fast pair mining.
        self._label_to_indices: dict[int, list[int]] = {}
        for idx in range(len(self._base)):
            _, label, _ = self._base[idx]
            self._label_to_indices.setdefault(label, []).append(idx)

        self._labels_list = list(self._label_to_indices.keys())
        assert len(self._labels_list) >= 2, (
            "Need >= 2 classes for contrastive training."
        )

    def __len__(self) -> int:
        return len(self._base)

    def _get_tensor(self, idx: int) -> torch.Tensor:
        image, _, _ = self._base[idx]
        tensor: torch.Tensor = self.transform(image)
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, label_a, _ = self._base[idx]
        img_a = self._get_tensor(idx)

        if random.random() < self.pos_fraction:
            # Positive: another image of the same class.
            same_indices = [i for i in self._label_to_indices[label_a] if i != idx]
            idx_b = random.choice(same_indices) if same_indices else idx
            pair_label = 1
        else:
            # Negative: an image from a different class.
            neg_label = random.choice(
                [lbl for lbl in self._labels_list if lbl != label_a]
            )
            idx_b = random.choice(self._label_to_indices[neg_label])
            pair_label = 0

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
    model: SiameseNeuralNetwork,
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
    val_ds = OxfordSiamesePairs("validation", transform=VAL_TF)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
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

    backbone = ResNetBackbone(pretrained=True)
    model = SiameseNeuralNetwork(
        backbone=backbone,
        embedding_dim=CONFIG["embedding_dim"],
        device=device,
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
    from PIL import Image

    backbone = ResNetBackbone(pretrained=False)
    model = SiameseNeuralNetwork(
        backbone=backbone, embedding_dim=CONFIG["embedding_dim"], device=device
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    img_a = Image.open(img_path_a).convert("RGB")
    img_b = Image.open(img_path_b).convert("RGB")
    score = model.similarity_score(img_a, img_b)
    print(f"Similarity: {score:.4f}  (range -1 to 1, higher = more similar)")


if __name__ == "__main__":
    train()
