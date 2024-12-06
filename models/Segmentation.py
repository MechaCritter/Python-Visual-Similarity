import logging
from enum import Enum
import os

# Insert the path to the root of the project
import sys
sys.path.insert(0, '/home/ais/Bachelorarbeit/similarity_metrics_of_images')

from torch.utils.data import DataLoader
import torch
from segmentation_models_pytorch import Unet, DeepLabV3, DeepLabV3Plus, UnetPlusPlus, PSPNet, PAN
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.metrics import IoU

from src.datasets import ExcavatorDataset
from src.utils import append_json_list
from src.losses import MultiClassDiceLoss

__all__ = ['UNetModel', 'DeepLabV3Model', 'UNetPlusPlusModel', 'DeepLabV3PlusModel', 'PSPNetModel', 'PyramidAttentionNetworkModel']

class BaseSegmentationModel:
    def __init__(self,
                 model_class: torch.nn.Module,
                 model_path: str = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 weight_decay: float = 1e-5,
                 metrics: list = None,
                 activation: str = None,
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 classes: int = 12,
                 lr: float = 1e-3,
                 raw_output: bool = False,
                 logger_name: str = 'SegmentationModel'):
        """
        Base class constructor for segmentation models.

        :param model_class: The model class (e.g., Unet, DeepLabV3, DeepLabV3Plus)
        :param model_path:
        :param criterion: default is DiceLoss
        :param optimizer: default is Adam
        :param metrics: default is IoU
        :param activation:
        :param encoder_name:
        :param encoder_weights:
        :param classes:
        :param lr:
        :param raw_output:
        """
        self._logger = logging.getLogger(logger_name)
        self.model = model_class
        self.encoder_name = encoder_name
        self.classes = classes
        self.activation = activation
        self.lr = lr

        if model_path:
            self.model.load_state_dict(state_dict=torch.load(model_path))
            self._logger.info(f"Model loaded from {model_path} with parameters: {self.model}")
        else:
            self.model = model_class(encoder_name=encoder_name,
                                     encoder_weights=encoder_weights,
                                     classes=classes,
                                     activation=activation)
            self._logger.info(f"""New model created with the following info:
                            - Encoder name: {self.encoder_name}
                            - Activation: {self.activation}
                            - Classes: {self.classes}""")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Device used for model: {self.device}")
        self.raw_output = raw_output
        self.model.to(self.device)

        self.criterion = criterion if criterion else MultiClassDiceLoss(mode='multiclass')
        self.metrics = metrics if metrics else [IoU()]
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

        self._train_epoch = TrainEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True
        )
        self._valid_epoch = ValidEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metrics,
            device=self.device,
            verbose=True
        )

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              decay_coeff: float = 0.96,
              log_save_path: str = None,
              model_save_path: str = 'models/torch_model_files/model.pt') -> None:
        """
        Train the model with the given data loaders.

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param num_epochs: number of epochs to train
        :param model_save_path: path to save the model
        """
        self._logger.info("Training model")
        if os.path.exists(model_save_path):
            raise FileExistsError(f"Model file already exists at {model_save_path}. Please delete it or specify a new path.")

        self._logger.info("""Training parameters:
        - Number of epochs: %s
        - Model save path: %s, 
        - Device: %s,
        - Criterion: %s,
        - Optimizer: %s,
        - Metrics: %s,
        - Activation: %s,
        - Encoder weights: %s,
        - Classes: %s,
        - Learning rate: %s
        """, num_epochs, model_save_path, self.device, self.criterion, self.optimizer, self.metrics, self.activation, self.encoder_name, self.classes, self.lr)
        max_score = 0
        for epoch in range(num_epochs):
            print('\nEpoch: {}'.format(epoch))
            train_logs = self._train_epoch.run(train_loader)
            valid_logs = self._valid_epoch.run(val_loader)

            # Save best model
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model.state_dict(), model_save_path)
                self._logger.info('Model saved at: %s', model_save_path)
            append_json_list(log_save_path, keyval={'train_dice_loss': [float(train_logs[self.criterion.__class__.__name__])],
                                                    'train_iou_score': [float(train_logs['iou_score'])],
                                                    'valid_dice_loss': [float(valid_logs[self.criterion.__class__.__name__])],
                                                    'valid_iou_score': [float(valid_logs['iou_score'])]})

            # Adjust learning rate
            if epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= decay_coeff
            self._logger.info("Current Learning Rate: %s", self.optimizer.param_groups[0]['lr'])
            self._logger.info("End of epoch %s. Currently, the best model achieves an IoU score of %s", epoch, max_score)

    def predict_single_image(self,
                             image: torch.Tensor,
                             gt_mask: torch.Tensor,
                             cls_index: int | Enum) -> tuple[float, torch.Tensor]:
        """
        Predict the prediction score and the mask of the given image.

        :param image: input image as a tensor. Shape. [1, C, H, W]
        :param gt_mask: ground truth mask as a tensor. Shape: [1, H, W]
        :param cls_index: index of class of interest. Can also be an Enum object

        :return: average confidence of the class of interest and the predicted mask
        """
        if not isinstance(cls_index, int):
            if not isinstance(cls_index, Enum):
                raise ValueError(f"cls_index must be an integer or an Enum object with integer value, got {type(cls_index)} instead.")
            else:
                self._logger.info("Enum object for class %s detected. Converting to integer %s", cls_index, cls_index:=cls_index.value)

        self.model.eval()
        image = image.to(self.device).unsqueeze(0)
        gt_mask = gt_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            raw_output = self.model(image)
            class_probs = raw_output[0, cls_index, :, :]
            mask_idx = (gt_mask.squeeze(0) == cls_index)

            total_confidence = class_probs[mask_idx].sum().item()
            num_pixels = mask_idx.sum().item()

            avr_confidence = total_confidence / num_pixels if num_pixels > 0 else 0

            if self.raw_output:
                return avr_confidence, raw_output.cpu().squeeze(0)

            _, predicted_mask = torch.max(raw_output, 1)
            return avr_confidence, predicted_mask.cpu().squeeze(0)

class UNetModel(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=Unet, logger_name='UNet', **kwargs)

class DeepLabV3Model(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=DeepLabV3, logger_name='DeepLabV3', **kwargs)

class UNetPlusPlusModel(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=UnetPlusPlus, logger_name='UNetPlusPlus', **kwargs)

class DeepLabV3PlusModel(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=DeepLabV3Plus, logger_name='DeepLabV3Plus', **kwargs)

class PSPNetModel(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=PSPNet, logger_name='PSPNet', **kwargs)

class PyramidAttentionNetworkModel(BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=PAN, logger_name='PyramidAttentionNetwork', **kwargs)


if __name__ =="__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from src.losses import FocalLoss, HybridFocalDiceLoss
    from src.utils import mask_to_rgb
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Define transformations
    train_transformer = A.Compose([
        # Transformations applied to both image and mask
        A.Resize(640, 640),  # Resize both
        A.Rotate(limit=30, p=0.5),  # Rotate both

        # Transformations applied only to the image
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),  # RGB shift
        # Scale pixel values of the image to [0, 1]
        A.Normalize(normalization='min_max_per_channel'),  # Image scaling

        # Convert image and mask to tensor
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    val_transformer = A.Compose([
        A.Resize(640, 640),
        A.Normalize(normalization='min_max_per_channel'),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    trainloader = DataLoader(ExcavatorDataset(transform=train_transformer,
                                              purpose='train',
                                              return_type='image+mask',
                                              one_hot_encode_mask=True
                                              ), batch_size=32, shuffle=True)
    validloader = DataLoader(ExcavatorDataset(transform=val_transformer,
                                              purpose='test',
                                              return_type='image+mask',
                                              one_hot_encode_mask=True
                                              ), batch_size=187, shuffle=False)

    hybrid_loss = HybridFocalDiceLoss(mode='multiclass', gamma=2.0, from_logits=False, ignore_index=None, dice_weight=0.3, focal_weight=0.7, normalize_weights=False, smooth=1e-5, eps=1e-7)

    dlv3 = DeepLabV3Model(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=1e-5, encoder_name='resnet18')
    dlv3plus = DeepLabV3PlusModel(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=1e-5, encoder_name='resnet18')
    unet = UNetModel(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=5e-6, encoder_name='resnet18')
    unetpp = UNetPlusPlusModel(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=5e-6, encoder_name='resnet18')
    pspnet = PSPNetModel(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=1e-5, encoder_name='resnet18')
    pan = PyramidAttentionNetworkModel(lr=0.02, activation='softmax2d', criterion=hybrid_loss, weight_decay=1e-5, encoder_name='resnet18')
    dlv3.train(trainloader,
               validloader,
               100,
               decay_coeff=0.96,
               log_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/res/dlv3_{hybrid_loss.__name__}_sum.json',
               model_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/models/torch_model_files/DeepLabV3_{hybrid_loss.__name__}.pt')
    unet.train(trainloader,
               validloader,
               100,
               decay_coeff=0.96,
                log_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/res/unet_{hybrid_loss.__name__}_sum.json',
               model_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/models/torch_model_files/UNet_{hybrid_loss.__name__}.pt')
    dlv3plus.train(trainloader,
                   validloader,
                   100,
                   decay_coeff=0.96,
                   log_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/res/dlv3plus_{hybrid_loss.__name__}_sum.json',
                   model_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/models/torch_model_files/DeepLabV3Plus_{hybrid_loss.__name__}.pt')
    unetpp.train(trainloader,
                    validloader,
                    100,
                    decay_coeff=0.96,
                    log_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/res/unetpp_{hybrid_loss.__name__}_sum.json',
                    model_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/models/torch_model_files/UNetPlusPlus_{hybrid_loss.__name__}.pt')
    pan.train(trainloader,
                    validloader,
                    100,
                    decay_coeff=0.96,
                    log_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/res/pan_{hybrid_loss.__name__}_sum.json',
                    model_save_path=f'/home/ais/Bachelorarbeit/similarity_metrics_of_images/models/torch_model_files/PyramidAttentionNetwork_{hybrid_loss.__name__}.pt')
