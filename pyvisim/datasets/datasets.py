import os
from typing import Optional
from platformdirs import user_cache_dir
from multiprocessing import Process
import logging

import cv2
import numpy as np
import requests
import scipy
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from pyvisim._config import *

setup_logging()

__all__ = ['OxfordFlowerDataset']

logger = logging.getLogger('Data_Set')

# Specific to the Oxford Flowers dataset
_DATASET_ROOT = os.path.join(user_cache_dir("pyvisim"), "oxford_flower_dataset")
_IMAGE_DIR = os.path.join(_DATASET_ROOT, "images/jpg")
_IMAGE_LABEL_FILE = os.path.join(_DATASET_ROOT, "labels.mat")
_SETID_FILE = os.path.join(_DATASET_ROOT, "setid.mat")
_FILES_FLOWER_DATA = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    "setid": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
}
OXFORD_NUM_IMAGES = 8189
NUM_TEST_IMG = 6149
NUM_TRAIN_IMG = 1020
NUM_VAL_IMG = 1020

def _download_and_process_file(url: str, dest: str, extract_dir: str):
    """
    Downloads a file and processes it (e.g., extraction if it's a zip or tar.gz file).
    """
    _download_file_with_progress(url, dest)

    if dest.endswith(".zip"):
        _extract_zip(dest, os.path.join(extract_dir, os.path.splitext(os.path.basename(dest))[0]))
        os.remove(dest)
    elif dest.endswith(".tgz") or dest.endswith(".tar.gz"):
        _extract_tar(dest, os.path.join(extract_dir, os.path.splitext(os.path.basename(dest))[0]))
        os.remove(dest)

def _download_file_with_progress(url: str, dest: str):
    """
    Download a file with a progress bar.
    """
    logger.info(f"Downloading from {url} to {dest}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(dest)}"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    logger.info(f"Downloaded file to {dest}")

def _extract_zip(zip_file: str, extract_to: str):
    """
    Extract a zip archive.
    """
    import zipfile
    logger.info(f"Extracting {zip_file} to {extract_to}")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        total_files = len(zf.infolist())
        with tqdm(total=total_files, unit='file', desc=f"Extracting {os.path.basename(zip_file)}") as progress_bar:
            for file in zf.infolist():
                zf.extract(file, extract_to)
                progress_bar.update(1)

def _extract_tar(tar_file: str, extract_to: str):
    """
    Extract a tar.gz archive.
    """
    import tarfile
    logger.info(f"Extracting {tar_file} to {extract_to}")
    with tarfile.open(tar_file, "r:gz") as tf:
        members = tf.getmembers()
        total_files = len(members)
        with tqdm(total=total_files, unit='file', desc=f"Extracting {os.path.basename(tar_file)}") as progress_bar:
            for member in members:
                tf.extract(member, path=extract_to)
                progress_bar.update(1)

def _data_downloaded() -> bool:
    """
    Check if the image files, labels, and setid files are downloaded.
    """
    if not os.path.isdir(_DATASET_ROOT):
        return False
    if not os.path.isdir(_IMAGE_DIR):
        return False
    if not os.path.isfile(_IMAGE_LABEL_FILE) or not os.path.isfile(_SETID_FILE):
        return False
    return True

def _check_data_integrity() -> bool:
    """
    Checks if the downloaded data are correct:
      1) labels.mat has exactly OXFORD_NUM_IMAGES labels
      2) setid.mat has correct lengths for tstid, valid, trnid
      3) images/ has exactly OXFORD_NUM_IMAGES images
    Returns True if all checks pass, False otherwise.
    """
    if not os.path.isfile(_IMAGE_LABEL_FILE):
        logger.warning(f"Label file not found at {_IMAGE_LABEL_FILE}.")
        return False
    try:
        mat_data = scipy.io.loadmat(_IMAGE_LABEL_FILE)
        labels = mat_data['labels'].squeeze().tolist()
        if len(labels) != OXFORD_NUM_IMAGES:
            logger.warning(f"Expected {OXFORD_NUM_IMAGES} labels, got {len(labels)}.")
            return False
    except Exception as e:
        logger.warning(f"Error reading labels file: {e}")
        return False

    if not os.path.isfile(_SETID_FILE):
        logger.warning(f"setid.mat not found at {_SETID_FILE}.")
        return False
    try:
        mat_data = scipy.io.loadmat(_SETID_FILE)
        tstid = mat_data['tstid'].squeeze().tolist()   # len 6149
        valid = mat_data['valid'].squeeze().tolist()   # len 1020
        trnid = mat_data['trnid'].squeeze().tolist()   # len 1020
        if len(tstid) != NUM_TEST_IMG or len(valid) != NUM_VAL_IMG or len(trnid) != NUM_TRAIN_IMG:
            logger.warning(f"setid.mat has incorrect lengths. tstid={len(tstid)}, "
                           f"valid={len(valid)}, trnid={len(trnid)}."
                           f"Expected {NUM_TEST_IMG}, {NUM_VAL_IMG}, {NUM_TRAIN_IMG} images respectively.")
            return False
    except Exception as e:
        logger.warning(f"Error reading setid file: {e}")
        return False

    if not os.path.isdir(_IMAGE_DIR):
        logger.warning(f"Image directory not found at {_IMAGE_DIR}.")
        return False
    jpgs = [f for f in os.listdir(_IMAGE_DIR) if f.lower().endswith('.jpg')]
    if len(jpgs) != OXFORD_NUM_IMAGES:
        logger.warning(f"Expected {OXFORD_NUM_IMAGES} .jpg images, got {len(jpgs)}.")
        return False
    return True

def download_oxford_flowers_data():
    """
    Downloads the 102 flowers dataset and organizes it into the desired structure,
    under `destination/oxford_flower_dataset/`.
    """
    logger.info("Starting download process for Oxford Flowers.")
    os.makedirs(_DATASET_ROOT, exist_ok=True)

    processes = []
    for name, url in _FILES_FLOWER_DATA.items():
        file_path = os.path.join(_DATASET_ROOT, f"{name}{os.path.splitext(url)[-1]}")
        p = Process(target=_download_and_process_file, args=(url, file_path, _DATASET_ROOT))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("Oxford Flowers dataset downloaded and processed successfully.")


class OxfordFlowerDataset(Dataset):
    """
    Oxford Flower Dataset. It can be found at: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html.

    In the original dataset, number of train images ('trnid') is 1020,
    number of validation images ('valid') is 1020, and number of test images ('tstid') is 6149. Since
    it makes more sense to have more images for training for this project, the train and test
    splits have been swapped.

    :param transform: Transformations to apply to the images.
    :param purpose: Purpose of the dataset ('train', 'test', 'validation'). You
    can also pass a list such as ['train', 'validation'] to get a combined dataset.
    """
    def __init__(self,
                 transform: Optional[transforms.Compose] = None,
                 purpose: str | list[str] = 'train') -> None:
        if transform is not None:
            raise NotImplementedError("Transformations are not yet supported.")
        self.transform = transform
        self.purpose = [purpose] if isinstance(purpose, str) else purpose
        if len(set(self.purpose)) < len(self.purpose):
            raise ValueError("Duplicate purposes found in the list. Please provide unique purposes.")
        if not _data_downloaded() or not _check_data_integrity():
            download_oxford_flowers_data()
        self.labels = self._load_labels(_IMAGE_LABEL_FILE)
        self.image_paths = self._load_image_paths()
        self.train_ids, self.val_ids, self.test_ids = self._load_set_ids(_SETID_FILE)
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
            [f for f in os.listdir(_IMAGE_DIR) if f.endswith('.jpg')]
        )
        return [os.path.join(_IMAGE_DIR, img) for img in images]

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

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, str]:
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

