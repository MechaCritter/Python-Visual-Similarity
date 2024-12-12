import logging
import logging.config
import pathlib
import torch

import yaml
from torchvision import transforms

# -Paths for the dataset- #
TRAIN_IMG_DATA_PATH_EXCAVATOR= r"D:\bachelor_thesis\excavator_dataset_w_masks\train" #TODO: change to relative paths
TRAIN_MASK_DATA_PATH_EXCAVATOR = r"D:\bachelor_thesis\excavator_dataset_w_masks\train_annot"
TEST_IMG_DATA_PATH_EXCAVATOR= r"D:\bachelor_thesis\excavator_dataset_w_masks\test"
TEST_MASK_DATA_PATH_EXCAVATOR = r"D:\bachelor_thesis\excavator_dataset_w_masks\test_annot"
VALID_IMG_DATA_PATH_EXCAVATOR= r"D:\bachelor_thesis\excavator_dataset_w_masks\validation"
VALID_MASK_DATA_PATH_EXCAVATOR = r"D:\bachelor_thesis\excavator_dataset_w_masks\validation_annot"

# -Config for the dataset- #
IMAGE_SIZE = (640, 640)
TRANSFORMER = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE)
])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = pathlib.Path(__file__).parent.parent

def setup_logging(default_path=rf"{ROOT}/res/logging_config.yaml", default_level=logging.INFO):
    """Setup logging configuration"""
    try:
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in Logging Configuration: {e}")
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
