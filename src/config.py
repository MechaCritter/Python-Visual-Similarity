import logging
import logging.config
import pathlib

import yaml
from torchvision import transforms
import torch

# -Config for the dataset- #
ROOT = pathlib.Path(__file__).parent.parent
IMAGE_SIZE = (640, 640)
TRANSFORMER = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE)
])

# - Device - #
ENFORCE_CUDA = True

def get_device():
    """Get device (if available)"""
    global ENFORCE_CUDA
    if ENFORCE_CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your computer's configuration.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()
print(f"Device used: {DEVICE}")

# -Paths for the excavator dataset- #
TRAIN_IMG_DATA_PATH_EXCAVATOR= rf"{ROOT}/excavator_dataset_w_masks/train"
TRAIN_MASK_DATA_PATH_EXCAVATOR = rf"{ROOT}/excavator_dataset_w_masks/train_annot"
TEST_IMG_DATA_PATH_EXCAVATOR = rf"{ROOT}/excavator_dataset_w_masks/test"
TEST_MASK_DATA_PATH_EXCAVATOR = rf"{ROOT}/excavator_dataset_w_masks/test_annot"
VALID_IMG_DATA_PATH_EXCAVATOR = None
VALID_MASK_DATA_PATH_EXCAVATOR = None

# -Paths for the Flower dataset- #
IMG_DATA_PATH_FLOWER = rf"{ROOT}/oxford_flower_dataset/images"
LABELS_PATH_FLOWER = rf"{ROOT}/oxford_flower_dataset/imagelabels.mat"
SETID_PATH_FLOWER = rf"{ROOT}/oxford_flower_dataset/setid.mat"


# - Logging - #
def setup_logging(default_path=rf"{ROOT}/res/logging_config.yaml", default_level=logging.INFO):
    """Setup logging configuration"""
    try:
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in Logging Configuration: {e}")
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


