"""
TODO: This module was made to compute loss of segmentation models.
TODO: For the future, implement loss for retrieval (e.g. triplet loss, contrastive loss, "similarity loss" etc.)
"""

import logging
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from src.utils import soft_dice_score

__all__ = ["MultiClassDiceLoss"]

class MultiClassDiceLoss(_Loss):
    __name__ = "MultiClassDiceLoss"
    def __init__(
                self,
                mode: str,
                classes: Optional[torch.Tensor] = None,
                log_loss: bool = False,
                from_logits: bool = True,
                smooth: float = 0.0,
                eps: float = 1e-7,
                ignore_index: Optional[int] = None) -> None:
        """
        Dice loss for image segmentation task.
        """
        super().__init__()
        assert mode in {'binary', 'multiclass'}, f"Unknown mode: {mode}. Supported modes are 'multiclass' and 'binary'."
        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param y_pred: (B, C, H, W)
        :param y_true: (B, C, H, W)

        :return: dice loss
        """
        assert y_true.dim() == y_pred.dim() == 4, f"Expected 4D input tensors, got {y_pred.dim()} for y_pred and {y_true.dim()} for y_true"
        logging.debug(f"""
                    Batch size: {y_true.size(0)}
                    Number of classes: {y_true.size(1)}
                    Image shape: ({y_true.size(2)} x {y_true.size(3)})
                    """)
        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = F.softmax(y_pred, dim=1)
            elif self.mode == 'binary':
                y_pred = F.sigmoid(y_pred)

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(batch_size, num_classes, -1)
        y_pred = y_pred.view(batch_size, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0 # Zeros out loss of classes that are not present in the mask (otherwise they would have a loss of 1, which is nonsense!
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


class FocalLoss(_Loss):
    __name__ = "FocalLoss"
    def __init__(self,
                 mode: str,
                 alpha: Optional[torch.Tensor] = None,
                 normalize_weights: bool = True,
                 gamma: float = 2.0,
                 from_logits: bool = True,
                 ignore_index: Optional[int] = None) -> None:
        """
        Focal loss for image segmentation task.
        """
        super().__init__()
        assert mode in {'binary', 'multiclass'}, f"Unknown mode: {mode}. Supported modes are 'multiclass' and 'binary'."
        self.mode = mode
        self.alpha = alpha
        if self.alpha is not None and normalize_weights:
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param y_pred: (B, C, H, W)
        :param y_true: (B, C, H, W)

        :return: Weighted focal loss
        """
        assert y_pred.dim() == y_pred.dim() == 4, f"Expected 4D input tensors, got {y_pred.dim()} for y_pred and {y_true.dim()} for y_true"
        y_true = torch.argmax(y_true, dim=1) # Revert one-hot encoding back to (B, H, W) tensor

        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = F.softmax(y_pred, dim=1)
            elif self.mode == 'binary':
                y_pred = torch.sigmoid(y_pred)

        if self.mode == 'multiclass':
            num_classes = y_pred.size(1)
            # Flatten predictions and targets
            y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)  # Shape: (N, C)
            y_true = y_true.view(-1)  # Shape: (N,)

            if self.ignore_index is not None:
                valid_mask = y_true != self.ignore_index
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]

            # Gather probabilities of the true class
            p_t = y_pred[torch.arange(y_pred.size(0)), y_true]

            # Handle alpha
            if self.alpha is None:
                self.alpha = torch.ones(num_classes, device=y_pred.device)/num_classes
            if not self.alpha.device == y_pred.device:
                self.alpha = self.alpha.to(y_pred.device)
            alpha_t = self.alpha[y_true]

        elif self.mode == 'binary':
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1).float()

            if self.ignore_index is not None:
                valid_mask = y_true != self.ignore_index
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]

            p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
            if self.alpha is not None:
                alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            else:
                alpha_t = 1.0

        # Compute focal loss
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * (-torch.log(p_t.clamp(min=1e-7)))

        return loss.mean()

class HybridFocalDiceLoss(_Loss):
    def __init__(self,
                 mode: str,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 from_logits: bool = True,
                 ignore_index: Optional[int] = None,
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 smooth: float = 1e-5,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(mode=mode,
                                    alpha=alpha,
                                    gamma=gamma,
                                    from_logits=from_logits,
                                    ignore_index=ignore_index)
        self.dice_loss = MultiClassDiceLoss(mode=mode,
                                            from_logits=from_logits,
                                            smooth=smooth,
                                            eps=eps)
        if not dice_weight + focal_weight == 1.0:
            raise ValueError(f"Sum of dice_weight and focal_weight must be equal to 1.0, got {dice_weight} + {focal_weight} = {dice_weight + focal_weight}")
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        loss = self.focal_weight * focal + self.dice_weight * dice
        return loss

if __name__ == "__main__":
    from src.datasets import ExcavatorDataset
    from src.utils import mask_to_rgb
    from models.Segmentation import DeepLabV3Model
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    dataset = ExcavatorDataset(transform=transformer,
                                              purpose='train',
                                              return_type='image+mask')
    trainloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='train',
                                              return_type='image+mask',
                                              one_hot_encode_mask=True
                                              ), batch_size=10, shuffle=False) # TODO: setting batch size above 1 won't work?
    validloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='validation',
                                              return_type='image+mask',
                                                one_hot_encode_mask=True
                                              ), batch_size=1, shuffle=False)
    model = DeepLabV3Model()
    model.model.load_state_dict(torch.load('models/torch_model_files/DeepLabV3_with_background.pt'))
    criterion = FocalLoss(mode='multiclass', from_logits=True)
    for i, (image, mask) in enumerate(trainloader):
        image = image.to('cuda')
        mask = mask.to('cuda')
        output = model.model(image)
        loss = criterion.forward(output, mask)
        print(loss)
        if i == 0:
            break


