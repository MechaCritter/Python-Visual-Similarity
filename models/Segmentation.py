import logging
import os

from torch.utils.data import DataLoader
import torch
from segmentation_models_pytorch import Unet, DeepLabV3, DeepLabV3Plus, UnetPlusPlus, PSPNet, PAN
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.metrics import IoU

from src.datasets import ExcavatorDataset
from src.utils import append_json_list
from src.losses import MultiClassDiceLoss


class _BaseSegmentationModel:
    def __init__(self,
                 model_class: torch.nn.Module,
                 model_path: str = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 weight_decay: float = 1e-5,
                 metrics: list = None,
                 activation: str = 'softmax',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 classes: int = 12,
                 lr: float = 1e-3,
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
        """
        self._logger = logging.getLogger(logger_name)
        self.model = model_class
        self.encoder_name = encoder_name
        self.classes = classes
        self.activation = activation
        self.lr = lr

        # Instantiate the model first
        self.model = model_class(encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 classes=classes,
                                 activation=activation)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            self._logger.info(f"""New {model_class} model created with the following info:
                            - Encoder name: {self.encoder_name}
                            - Activation: {self.activation}
                            - Classes: {self.classes}""")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Device used for model: {self.device}")
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
        :param num_epochs: number of epochs
        :param decay_coeff: lr is multiplied by this value after each epoch
        :param log_save_path: path to save the training logs
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
            if log_save_path:
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
                             cls_of_interest: list[int] = None,
                             raw_output: bool = True,
                             mean: bool = True,
                             return_raw_prob_vector: bool = False,
                             ignore_background: bool = False
                             ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the prediction score and the mask of the given image. This method assumes that
        '0' is the background class.

        :param image: input image as a tensor. Shape. [1, C, H, W]
        :param gt_mask: ground truth mask as a tensor. Shape: [1, H, W]
        :param cls_of_interest: index of classes of interest.
        :param raw_output: if False, one-hot encode the output mask
        :param mean: if False, return the confidence of each class
        :param return_raw_prob_vector: if True, returns a raw probability vector for all classes (or all but background).
                                       In this case, cls_of_interest and mean are ignored.
        :param ignore_background: if True and return_raw_prob_vector=True, background class is omitted from the returned vector.
        :return: (confidence, pred_mask):
            - If return_raw_prob_vector=True:
                confidence is a vector of shape (num_classes,) or (num_classes-1) if ignore_background=True.
            - Else:
                confidence is either a single mean confidence value or a vector of confidences depending on `mean` and `cls_of_interest`.
            pred_mask is either the raw probability mask or the argmax mask depending on `raw_output`.

        :raises ValueError: if cls_of_interest is not an integer
        """
        if gt_mask.shape != image.shape[-2:]:
            raise ValueError(f"Ground truth mask shape must be the same as the image shape. "
                             "Also, mask has to be a single channel grayscale image. "
                             f"Got {gt_mask.shape} instead of {(1, *image.shape[-2:])}")

        self.model.eval()
        image = image.to(self.device).unsqueeze(0)
        gt_mask = gt_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred_mask = self.model(image)
            if return_raw_prob_vector:
                probs = pred_mask.squeeze(0) # (C, H, W)
                if cls_of_interest is not None and len(cls_of_interest) > 0:
                    class_indices = torch.tensor(cls_of_interest, device=self.device)
                else:
                    if ignore_background:
                        class_indices = torch.arange(1, self.classes, device=self.device)
                    else:
                        class_indices = torch.arange(self.classes, device=self.device)

                cls_probs = torch.tensor([probs[c, :, :].mean().item() for c in class_indices], device=self.device, dtype=torch.float32)

                if not raw_output:
                    _, pred_mask = torch.max(pred_mask, 1)
                    return cls_probs, pred_mask.cpu().squeeze(0)

                return cls_probs, pred_mask.cpu().squeeze(0)

            if cls_of_interest is not None:
                if not all(isinstance(cls, int) for cls in cls_of_interest):
                    raise ValueError(f"cls_of_interest must be an integer, "
                                     f"got {type(cls_of_interest)} instead.")
                if len(cls_of_interest) > self.classes:
                    raise ValueError(
                        f"Number of classes of interest must be less than or equal to the number of classes in the model. "
                        f"Got {len(cls_of_interest)} classes of interest for a model with {self.classes} classes.")
                unique_cls = torch.tensor(cls_of_interest, device=self.device)
            else:
                unique_cls = torch.unique(gt_mask)

            confidence = torch.zeros(len(unique_cls), device=self.device, dtype=torch.float32)
            for i, c in enumerate(unique_cls):
                mask_idx = (gt_mask.squeeze(0) == unique_cls[c])
                class_probs = pred_mask[0, unique_cls[c], :, :]
                total_confidence = class_probs[mask_idx].sum().item()
                num_pixels = mask_idx.sum().item()
                avr_confidence = total_confidence / num_pixels if num_pixels > 0 else 0
                confidence[c] = avr_confidence

            if mean:
                confidence = confidence.mean()

            if not raw_output:
                _, pred_mask = torch.max(pred_mask, 1)

            return confidence, pred_mask.cpu().squeeze(0)

class UNetModel(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=Unet, logger_name='UNet', **kwargs)

class DeepLabV3Model(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=DeepLabV3, logger_name='DeepLabV3', **kwargs)

class UNetPlusPlusModel(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=UnetPlusPlus, logger_name='UNetPlusPlus', **kwargs)

class DeepLabV3PlusModel(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=DeepLabV3Plus, logger_name='DeepLabV3Plus', **kwargs)

class PSPNetModel(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=PSPNet, logger_name='PSPNet', **kwargs)

class PyramidAttentionNetworkModel(_BaseSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(model_class=PAN, logger_name='PyramidAttentionNetwork', **kwargs)


if __name__ =="__main__":
    from src.datasets import ExcavatorDataset
    import matplotlib.pyplot as plt
    from src.config import TRANSFORMER, ROOT
    from src.utils import mask_to_rgb, get_class_idx

    def plot_image_and_mask(image: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor=None):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Image')
        plt.subplot(1, 3, 2)
        if gt_mask.shape[0] == 3:
            gt_mask = gt_mask.permute(1, 2, 0)
        plt.imshow(gt_mask)
        plt.axis('off')
        plt.title('Ground Truth Mask')

        if pred_mask is None:
            plt.show()
            return

        plt.subplot(1, 3, 3)
        if pred_mask.shape[0] == 3:
            pred_mask = pred_mask.permute(1, 2, 0)
        plt.imshow(pred_mask)
        plt.axis('off')
        plt.title('Predicted Mask')
        plt.show()

    dataset = ExcavatorDataset(plot=True, purpose='train', return_type='image+mask', transform=TRANSFORMER)
    test_dataset = ExcavatorDataset(plot=True, purpose='test', return_type='image+mask', transform=TRANSFORMER)
    img, mask = dataset[40]
    test_img, test_mask = test_dataset[40]
    classes = get_class_idx(mask, class_colors=dataset.class_colors)
    classes_test = get_class_idx(test_mask, class_colors=dataset.class_colors)
    print("Unique classes in training image:", classes)
    print("Unique classes in test image:", classes_test)
    dlv3 = DeepLabV3Model(model_path=f'{ROOT}/models/torch_model_files/DeepLabV3_HybridFocalDiceLoss.pt')

    confidence, pred_mask = dlv3.predict_single_image(img, mask, return_raw_prob_vector=True, cls_of_interest=classes, raw_output=False)
    confidence_test, pred_mask_test = dlv3.predict_single_image(test_img, test_mask, return_raw_prob_vector=True, cls_of_interest=classes, raw_output=False)
    pred_mask = mask_to_rgb(pred_mask, class_colors=dataset.class_colors)
    pred_mask_test = mask_to_rgb(pred_mask_test, class_colors=dataset.class_colors)
    print("Confidence in training image:", confidence)
    print("Confidence in test image:", confidence_test)
    rgb_mask = mask_to_rgb(mask, class_colors=dataset.class_colors)
    rgb_mask_test = mask_to_rgb(test_mask, class_colors=dataset.class_colors)

    plot_image_and_mask(img, rgb_mask, pred_mask)
    plot_image_and_mask(test_img, rgb_mask_test, pred_mask_test)
