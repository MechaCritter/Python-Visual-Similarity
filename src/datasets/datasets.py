import os
from enum import Enum
from typing import Optional

import albumentations
import cv2
import scipy
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import *
from src.utils import rgb_to_mask, plot_image, permute_image_channels

setup_logging()

__all__ = ['OxfordFlowerDataset']

class BaseDataset(Dataset):
    """
    **Note**: this dataset does not support slice-wise indexing. To load multiple images, use a `dataloader` instead.

    Base class for datasets used in this project. The folder structure of all child classes should be as follows:

    ```
    data
    ├── train
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── test
    │   │── image1.jpg
    │   │── image2.jpg
    │   │── ...
    │   │── image1.jpg
    │   │── image2.jpg
    │   │── ...

    ```
    """
    def __init__(self,
                 train_img_data_dir: str = None,
                 validation_img_data_dir: Optional[str] = None,
                 test_img_data_dir: str = None,
                 train_mask_data_dir: Optional[str] = None,
                 validation_mask_data_dir: Optional[str] = None,
                 test_mask_data_dir: Optional[str] = None,
                 *,
                 transform: transforms = None,
                 one_hot_encode_mask: bool = False,
                 plot: bool = False,
                 verbose: bool = False,
                 purpose: str = 'train',
                 return_type: str = 'image+mask',
                 class_colors: dict = None,
                 mask_suffix: str = "_mask",
                 num_classes: int = None) -> None:
        """
        Class constructor. Remember to pass data to whether train or test data directory.
        If only one path is provided, it is passed to the train dataset.

        :param train_img_data_dir: Path to the train data directory
        :param validation_img_data_dir: Path to the validation data directory (optional)
        :param test_img_data_dir: Path to the test data directory
        :param train_mask_data_dir: Path to the train mask data directory (for segmentation tasks)
        :param validation_mask_data_dir: Path to the validation mask data directory (for segmentation tasks)
        :param test_mask_data_dir: Path to the test mask data directory (for segmentation tasks)
        :param transform: Transformation to apply to the images
        :param one_hot_encode_mask: Whether to one-hot encode the mask to shape (num_classes, height, width)
        :param return_type: Whether to return the image, mask, label or all
        :param plot: Whether to plot the images
        :param verbose: Whether to print out extra information for debugging
        :param purpose: Whether the data is for training, validation or testing
        :param mask_suffix: when passed, masks are expected to have name `{image_name}{mask_suffix}.png`
        :param class_colors: Dictionary containing the class colors for the dataset for segmentation tasks

        :raises FileNotFoundError: If 'train_data_dir' or 'test_data_dir' does not exist
        """
        self._logger = logging.getLogger('Data_Set')
        self._logger.name = self.__class__.__name__
        self.max_img_to_plot: int = 10
        self.verbose: bool = verbose
        self.one_hot_encode_mask: bool = one_hot_encode_mask
        self.plot: bool = plot
        self.return_type: str = return_type
        self.purpose: str = purpose
        self._class_colors: dict = class_colors
        self.num_classes = num_classes
        self.mask_suffix = mask_suffix

        self._logger.debug(f"Initializing BaseDataset with purpose: {self.purpose}")
        self._logger.debug(f"Train image data directory: {train_img_data_dir}")
        self._logger.debug(f"Validation image data directory: {validation_img_data_dir}")
        self._logger.debug(f"Test image data directory: {test_img_data_dir}")

        if not os.path.exists(train_img_data_dir):
            raise FileNotFoundError(f"Directory {train_img_data_dir} does not exist.")

        self.train_img_data_dir = train_img_data_dir
        self.validation_img_data_dir = validation_img_data_dir
        self.test_img_data_dir = test_img_data_dir
        self.train_mask_data_dir = train_mask_data_dir
        self.validation_mask_data_dir = validation_mask_data_dir
        self.test_mask_data_dir = test_mask_data_dir

        self.images = []
        self.masks = []

        self.transform: transforms = transform

        # Method calls
        self.load_images()

    @property
    def class_colors(self) -> dict:
        return self._class_colors

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.

        :raises ValueError: If the purpose is not 'train', 'validation' or 'test'
        """
        match self.purpose:
            case 'train':
                self._load_from_dir(self.train_img_data_dir, annot_data_dir=self.train_mask_data_dir)
            case 'validation':
                self._load_from_dir(self.validation_img_data_dir, annot_data_dir=self.validation_mask_data_dir)
            case 'test':
                self._load_from_dir(self.test_img_data_dir, annot_data_dir=self.test_mask_data_dir)
            case _:
                raise ValueError(f"Purpose has to be 'train', 'validation' or 'test'.")
        if self.verbose:
            self._logger.debug("Loaded %s images with purpose '%s'", len(self.images), self.purpose)

    def _load_from_dir(self, data_dir: str, annot_data_dir: str=None) -> None:
        """
        Internal method used to pass the data directory and return a list that contain dictionaries of the image path and label.

        :param data_dir: Path to the directory containing the images. Pass the path to the `data/train`and `data/test` directories. The names
        of the subdirectories are used as labels.

        :return: A list of path to the images and their labels
        """
        image_files = os.listdir(data_dir)
        if not image_files:
            raise FileNotFoundError(f"Directory {data_dir} is empty.")
        for img_file in image_files:
            if not img_file.endswith(".jpg"):
                self._logger.warning(f"Skipping file/folder {img_file}. Only .jpg files are supported.")
            image_path = os.path.join(data_dir, img_file)
            self.images.append(image_path)
            if annot_data_dir:
                mask_file = img_file.replace(".jpg", f"{self.mask_suffix}.png")
                if not os.path.exists(mask_path:=os.path.join(annot_data_dir, mask_file)):
                    raise FileNotFoundError(f"Mask file not found for image {image_path}. Expected path: {mask_path}")
                self.masks.append(mask_path)

    def __getitem__(self, index: int) -> tuple:
        """
        Get the image, mask, and label at the specified index. If `plot` is set to True, `plot_image` is called.

        :param index: Index of the image to retrieve

        :return: Image, mask, label and path, depending on the `return_type`

        :raises IndexError: If the index is out of range
        :raises ValueError: If `return_type` is invalid or slicing is used
        :raises FileNotFoundError: If the image or mask file is not found
        :raises RuntimeError: If the mask file is not a .png file, or not found
        """
        self._logger.debug(f"Retrieving item at index: {index} with return type: {self.return_type}")

        if isinstance(index, slice):
            raise ValueError("Slicing is not supported for this dataset. Use a dataloader instead.")
        elif index >= len(self.images) or index < 0:
            raise IndexError(f"Index out of range. Data for {self.purpose} purpose only contains {len(self.images)} images.")

        if not self.return_type in ['image', 'image+mask', 'image+mask+path', 'image+path']:
            raise ValueError(f"`return_type` has to be whether 'image', 'image+mask', 'image+mask+path' or 'image+path'. not {self.return_type}")

        image_path = self.images[index]
        mask_path = self.masks[index] if self.masks[index] else None

        image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if image_array is None:
            raise FileNotFoundError(f"Image file {image_path} could not be loaded.")

        mask = None
        if mask_path:
            if not mask_path.endswith(".png"):
                raise RuntimeError(f"Mask file {mask_path} is not a .png file. Only .png files are supported for masks.")
            mask = cv2.cvtColor(cv2.imread(mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                image_array = self.transform(image_array)
                if mask is not None:
                    mask = self.transform(mask)
            elif isinstance(self.transform, albumentations.core.composition.Compose):
                transformed = self.transform(image=image_array, mask=mask)
                image_array = transformed['image']
                if mask is not None:
                    mask = (transformed['mask'] / 255.0).float()

            if self._class_colors:
                self._logger.info("RGB mask detected with shape: %s. Converting to class mask.", mask.shape)
                if mask.shape[2] == 3 and len(mask.shape) == 3:
                    self._logger.info(f"Permuting image channels for mask with shape: {mask.shape}. New shape would be (C, H, W)")
                    mask = permute_image_channels(mask)
                mask = rgb_to_mask(mask, self._class_colors)
                self._logger.info("Mask converted with new shape: %s", mask.shape)
            if self.one_hot_encode_mask:
                mask = F.one_hot(mask.long(), num_classes=self.num_classes).permute(2, 0, 1).float()
            if self.plot:
                plot_image(image_array, title=f"Image: {index}")

        match self.return_type:
            case 'image':
                return image_array
            case 'image+mask':
                if mask is None:
                    raise RuntimeError(f"Mask for image {image_path} is None.")
                return image_array, mask
            case 'image+mask+path':
                if mask is None:
                    raise RuntimeError(f"Mask for image {image_path} is None.")
                return image_array, mask, image_path
            case 'image+path':
                return image_array, image_path

    def __len__(self) -> int:
        return len(self.images)


class Excavators(Enum):
    BACKGROUND = 0
    BULLDOZER = 1
    CAR = 2
    CATERPILLAR = 3
    CRANE = 4
    CRUSHER = 5
    DRILLER = 6
    EXCAVATOR = 7
    HUMAN = 8
    ROLLER = 9
    TRACTOR = 10
    TRUCK = 11


class ExcavatorDataset(BaseDataset):
    """
    **Note**: this dataset does not support slice-wise indexing. To load multiple images, use a `dataloader` instead.
    """
    def __init__(self,
                 train_img_data_dir: str = TRAIN_IMG_DATA_PATH_EXCAVATOR,
                    validation_img_data_dir: Optional[str] = VALID_IMG_DATA_PATH_EXCAVATOR,
                    test_img_data_dir: str = TEST_IMG_DATA_PATH_EXCAVATOR,
                    train_mask_data_dir: Optional[str] = TRAIN_MASK_DATA_PATH_EXCAVATOR,
                    validation_mask_data_dir: Optional[str] = VALID_MASK_DATA_PATH_EXCAVATOR,
                    test_mask_data_dir: Optional[str] = TEST_MASK_DATA_PATH_EXCAVATOR,
                    *,
                 transform: transforms = None,
                 one_hot_encode_mask: bool = False,
                 plot: bool = False,
                 verbose: bool = False,
                 purpose: str = 'train',
                 return_type: str = 'image+mask',
                 class_colors: dict = None) -> None:
        super().__init__(train_img_data_dir=train_img_data_dir,
                         train_mask_data_dir=train_mask_data_dir,
                         test_img_data_dir=test_img_data_dir,
                         test_mask_data_dir=test_mask_data_dir,
                         validation_img_data_dir=validation_img_data_dir,
                        validation_mask_data_dir=validation_mask_data_dir,
                         transform=transform,
                         one_hot_encode_mask=one_hot_encode_mask,
                         plot=plot,
                         verbose=verbose,
                         purpose=purpose,
                         return_type=return_type,
                         class_colors=class_colors,
                         num_classes=len(Excavators))
        self._class_colors = { # RGB colors for each class
            Excavators.BACKGROUND: torch.tensor([0, 0, 0]),
            Excavators.BULLDOZER: torch.tensor([235, 183, 0]),
            Excavators.CAR: torch.tensor([0, 255, 255]),
            Excavators.CATERPILLAR: torch.tensor([235, 16, 0]),
            Excavators.CRANE: torch.tensor([0, 252, 199]),
            Excavators.CRUSHER: torch.tensor([140, 0, 255]),
            Excavators.DRILLER: torch.tensor([254, 122, 14]),
            Excavators.EXCAVATOR: torch.tensor([171, 171, 255]),
            Excavators.HUMAN: torch.tensor([86, 0, 254]),
            Excavators.ROLLER: torch.tensor([255, 0, 255]),
            Excavators.TRACTOR: torch.tensor([0, 128, 128]),
            Excavators.TRUCK: torch.tensor([255, 34, 134]),
        }
        # Normalize and convert to tensors
        self._class_colors = {
            key: value.to(torch.float32) / 255.0
            for key, value in self._class_colors.items()
        }

class OxfordFlowerDataset(Dataset):
    """
    Oxford Flower Dataset. It can be found at: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

    Organize the data like this:

    ```
    oxford_flower_dataset/
    ├── images
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   └── ...
    │   ├── image_08189.jpg
    ├── imagelabels.mat
    └── setid.mat
    ```
    In the original dataset, number of train images ('trnid') is 1020,
    number of validation images ('valid') is 1020, and number of test images ('tstid') is 6149. Since
    it makes more sense to have more images for training for this project, the train and test
    splits have been swapped.
    
    :param image_dir: Directory containing image files.
    :param image_labels_file: Path to the file containing image labels.
    :param set_id_file: Path to the file containing set IDs (train/test/val splits).
    :param transform: Transformations to apply to the images.
    :param purpose: Purpose of the dataset ('train', 'test', 'validation'). You
    can also pass a list such as ['train', 'validation'] to get a combined dataset.
    """
    def __init__(self,
                 image_dir: str = IMG_DATA_PATH_FLOWER,
                 image_labels_file: str = LABELS_PATH_FLOWER,
                 set_id_file: str = SETID_PATH_FLOWER,
                 transform: Optional[transforms.Compose] = None,
                 purpose: str | list[str] = 'train') -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.purpose = [purpose] if isinstance(purpose, str) else purpose
        if len(set(self.purpose)) < len(self.purpose):
            raise ValueError("Duplicate purposes found in the list. Please provide unique purposes.")
        self.labels = self._load_labels(image_labels_file)
        self.image_paths = self._load_image_paths()
        self.train_ids, self.val_ids, self.test_ids = self._load_set_ids(set_id_file)
        self.image_paths, self.labels = self._filter_by_purpose()

    def _load_labels(self, labels_file: str) -> list[int]:
        """
        Load image labels from the given .mat file.

        :param labels_file: Path to the .mat file with labels.
        :return: List of labels.
        """
        mat_data = scipy.io.loadmat(labels_file)
        return mat_data['labels'].squeeze().tolist()

    def _load_image_paths(self) -> list[str]:
        """
        Get sorted paths to all images in the directory.

        :return: List of sorted image file paths.
        """
        images = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        )
        return [os.path.join(self.image_dir, img) for img in images]

    def _load_set_ids(self, set_id_file: str) -> tuple[list[int], list[int], list[int]]:
        """
        Load train, validation, and test IDs from the setid.mat file.

        :param set_id_file: Path to the .mat file with set IDs.
        :return: Tuple of train, validation, and test IDs.
        """
        mat_data = scipy.io.loadmat(set_id_file)
        train_ids = mat_data['tstid'].squeeze().tolist() # Swaps train and test, since test contains significantly more images
        val_ids = mat_data['valid'].squeeze().tolist()
        test_ids = mat_data['trnid'].squeeze().tolist()
        return train_ids, val_ids, test_ids

    def _filter_by_purpose(self) -> tuple[list[str], list[int]]:
        """
        Filter images and labels based on the dataset purpose.

        :return: Filtered image paths and labels.
        """
        chosen_ids = []
        for p in self.purpose:
            match p:
                case 'train':
                    chosen_ids += self.train_ids
                case 'validation':
                    chosen_ids += self.val_ids
                case 'test':
                    chosen_ids += self.test_ids
                case _:
                    raise ValueError(f"Unknown purpose: {p}. Must be 'train', 'validation', or 'test'.")

        chosen_ids = list(set(chosen_ids))

        filtered_paths = [self.image_paths[i - 1] for i in chosen_ids]
        filtered_labels = [self.labels[i - 1] for i in chosen_ids]
        return filtered_paths, filtered_labels

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        :return: Length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Get an image and its corresponding label.

        :param idx: Index of the image.
        :return: Tuple of transformed image, label, and image path.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx] if self.labels else -1

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

