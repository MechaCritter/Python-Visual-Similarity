from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Type
import json

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import transforms
from tqdm import tqdm
from piq import ssim, multi_scale_ssim as ms_ssim

from src.metrics import *
from src.utils import *
from models.Segmentation import UNetModel, DeepLabV3Model
from src.datasets import BaseDataset
from src.config import TRANSFORMER

# TODO: add a function to compute model outputs, then remove the model predictions from the process_image function
# TODO: Then, use the function to output and save model outputs as hdf5 files

# TODO: bring the function to compute VLAD and fisher vectors from notebook `calc_and_save_ssim_vlad_fisher.ipynb` to this file

def compute_and_save_ssim_matrices(dataset: BaseDataset,
                                   output_dir: str,
                                   batch_size: int = 1,
                                   gaussian_sigma: float = None,
                                   compression_quality: float = None) -> None:
    """
    Compute and save SSIM and MS-SSIM matrices for the given dataset. Then, save the result as a .pt file.
    In the output matrix for ssim and ms_ssim, only the upper triangular part is saved (since
    the matrix is symmetric).

    The kernel size is determined using the empirical rule:
    - The kernel radius should span 3 times the standard deviation: kernel_radius = int(3 * sigma)
    - The kernel size is 2 * kernel_radius + 1

    :param dataset: dataset object
    :param output_dir: output directory for the .pt files
    :param batch_size: batch size
    :param gaussian#_sigma: standard deviation for Gaussian blur
    :param compression_quality: quality for image compression

    :raises ValueError: if dataset is not transformed using TRANSFORMER from config.py
    """
    if dataset.transform != TRANSFORMER:
        raise ValueError("Please transform your dataset using the TRANSFORMER from config.py")

    num_imgs = len(dataset)
    kernel_size = None
    images = [image_array for image_array, *_ in dataset]
    image_paths = [path.replace('/', '|').replace('\\', '|') for *_, path in dataset]

    if gaussian_sigma:
        kernel_size = 2 * int(3 * gaussian_sigma) + 1  # Empirical rule
        print(f"Kernel size used for sigma={gaussian_sigma}: {kernel_size}")
        images = [gaussian_blur(img, sigma=gaussian_sigma, kernel_size=kernel_size) for img in images]

    if compression_quality:
        images = [compress_image(img, quality=compression_quality) for img in images]

    images = torch.stack(images, dim=0).to('cuda')

    indices = torch.triu_indices(num_imgs, num_imgs, offset=1, device='cuda')
    rows, cols = indices[0], indices[1]
    num_pairs = rows.size(0)
    ssim_result = torch.zeros((num_imgs, num_imgs), device='cuda')
    ms_ssim_result = torch.zeros((num_imgs, num_imgs), device='cuda')
    for batch_start in tqdm(range(0, num_pairs, batch_size), desc=f'Computing SSIM/MS-SSIM:'):
        batch_end = min(batch_start + batch_size, num_pairs)
        batch_rows = rows[batch_start:batch_end]
        batch_cols = cols[batch_start:batch_end]
        img_rows = images[batch_rows]
        img_cols = images[batch_cols]
        ssim_score = ssim(img_rows, img_cols, data_range=1.0, reduction='none')
        ms_ssim_score = ms_ssim(img_rows, img_cols, data_range=1.0, reduction='none')

        ssim_result[batch_rows, batch_cols] = ssim_score
        ms_ssim_result[batch_rows, batch_cols] = ms_ssim_score

    # torch.save(ssim_result, f'{output_dir}/ssim_sigma{sigma}.pt')
    # torch.save(ms_ssim_result, f'{output_dir}/ms_ssim_sigma{sigma}.pt')
    save_to_hdf5(f'{output_dir}/ssim_sigma{gaussian_sigma}.h5',
                 {'ssim': ssim_result.cpu().numpy(),
                            'ms_ssim': ms_ssim_result.cpu().numpy(),
                            'image_paths': image_paths})
    print("Saved SSIM and MS-SSIM matrices using: \n"
          f"sigma={gaussian_sigma}, kernel_size={kernel_size}, compression_quality={compression_quality}")











