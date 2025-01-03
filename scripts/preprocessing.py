import os
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import cv2  # Added for connected components analysis

from src.config import (TRAIN_MASK_DATA_PATH_EXCAVATOR, TEST_MASK_DATA_PATH_EXCAVATOR,
                        VALID_MASK_DATA_PATH_EXCAVATOR)

# Define the class colors and their corresponding names
CLASS_COLORS = {
    (235, 183, 0): 'bulldozer',
    (0, 255, 255): 'car',
    (235, 16, 0): 'caterpillar',
    (0, 252, 199): 'crane',
    (140, 0, 255): 'crusher',
    (254, 122, 14): 'driller',
    (171, 171, 255): 'excavator',
    (86, 0, 254): 'human',
    (255, 0, 255): 'roller',
    (0, 128, 128): 'tractor',
    (255, 34, 134): 'truck',
}

# Include the background color at index 0
CLASS_COLORS_LIST = [
    (0, 0, 0),  # Background
    (235, 183, 0),  # bulldozer
    (0, 255, 255),  # car
    (235, 16, 0),  # caterpillar
    (0, 252, 199),  # crane
    (140, 0, 255),  # crusher
    (254, 122, 14),  # driller
    (171, 171, 255),  # excavator
    (86, 0, 254),  # human
    (255, 0, 255),  # roller
    (0, 128, 128),  # tractor
    (255, 34, 134),  # truck
]

def denoise_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Denoises the input mask image by removing components smaller than the specified minimum size.

    :param mask: Input grayscale mask image where each unique pixel value represents a different class.
    :param min_size: Minimum pixel size for components to retain.

    :returns: Processed mask with only components of size >= min_size, retaining original class values.
    """
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    for class_value in np.unique(mask):
        if class_value == 0:  # Skip the background class
            continue

        # Create a binary mask for the current class
        class_mask = np.where(mask == class_value, 255, 0).astype(np.uint8)

        # Perform connected components analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=4)

        # Iterate over each component
        for i in range(1, num_labels):
            component_size = stats[i, cv2.CC_STAT_AREA]
            if component_size >= min_size:
                # Retain the component in the filtered mask
                filtered_mask[labels == i] = class_value

    return filtered_mask


def convert_mask_to_rgb(mask_path: str, min_size: int=100) -> None:
    """
    Converts a grayscale mask image into an RGB representation based on predefined class colors,
    after removing noise using a minimum component size threshold. Then, overwrites the original mask image
    with the denoised one.

    :param mask_path: Path to the input grayscale mask image.
    :param min_size: Minimum pixel size for retaining connected components in the mask.
    :returns: None; the processed RGB image overwrites the original mask image.
    """
    # Open the grayscale mask image
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)  # Shape: (H, W), values from 0-11

    # Denoise the mask
    logging.debug(f"Processing image {os.path.basename(mask_path)}")
    logging.debug("Number of classes in image %s before denoising: %d", mask_path, len(np.unique(mask_array)))
    denoised_mask_array = denoise_mask(mask_array, min_size)
    logging.debug("Number of classes in image %s after denoising: %d", mask_path, len(np.unique(denoised_mask_array)))

    # Create a lookup table (LUT) for class indices to colors
    LUT = np.array(CLASS_COLORS_LIST, dtype=np.uint8)

    # Map each index in the denoised mask to its corresponding RGB color
    rgb_array = LUT[denoised_mask_array]

    # Convert the RGB array back to an image
    rgb_image = Image.fromarray(rgb_array)

    # Overwrite the original grayscale image with the RGB image
    rgb_image.save(mask_path)

def convert_all_masks_to_rgb(min_size: int = 100) -> None:
    """
    Processes all grayscale masks in the specified directories and converts them
    to RGB format using the predefined class colors. The denoising step is
    applied to remove small components below the given size threshold.

    :param min_size: Minimum pixel size to retain connected components in the mask.
    """
    folders = [
        TRAIN_MASK_DATA_PATH_EXCAVATOR,
        TEST_MASK_DATA_PATH_EXCAVATOR,
        VALID_MASK_DATA_PATH_EXCAVATOR
    ]

    for mask_folder in folders:
        for mask_file in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_file)
            convert_mask_to_rgb(mask_path, min_size)


# def process_image(image_path, mask_path, output_image_dir, output_mask_dir):
#     try:
#         # Read the mask image and convert it to RGB
#         mask_image = Image.open(mask_path).convert('RGB')
#         mask_array = np.array(mask_image)
#         # Reshape the mask array to a list of RGB tuples
#         pixels = mask_array.reshape(-1, mask_array.shape[-1])
#         # Get unique colors and their counts in the mask
#         unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
#         # Map colors to classes and sum counts per class
#         class_counts = {}
#         for color, count in zip(unique_pixels, counts):
#             color_tuple = tuple(color)
#             if color_tuple in CLASS_COLORS:
#                 class_name = CLASS_COLORS[color_tuple]
#                 class_counts[class_name] = class_counts.get(class_name, 0) + count
#         if not class_counts:
#             return  # Skip if no classes are found
#         # Select the class with the maximum pixel count
#         logging.info(f"Class count for {image_path}: {class_counts}")
#         selected_class = max(class_counts, key=class_counts.get)
#         logging.info(f"Selected class for {image_path}: {selected_class}")
#         # Copy the image and mask to the corresponding class folders
#         class_image_dir = os.path.join(output_image_dir, selected_class)
#         class_mask_dir = os.path.join(output_mask_dir, selected_class)
#         os.makedirs(class_image_dir, exist_ok=True)
#         os.makedirs(class_mask_dir, exist_ok=True)
#         image_filename = os.path.basename(image_path)
#         mask_filename = os.path.basename(mask_path)
#         shutil.copy(image_path, os.path.join(class_image_dir, image_filename))
#         shutil.copy(mask_path, os.path.join(class_mask_dir, mask_filename))
#     except Exception as e:
#         logging.error(f"Error processing {image_path}: {e}")
#         raise e
#
#
# def main():
#     root_dir = r"D:\bachelor_thesis\excavator_dataset_w_masks"
#     min_size = 100  # Adjust the minimum size threshold as needed
#
#     # First, denoise and convert all masks to RGB
#     convert_all_masks_to_rgb(min_size)
#
#     folders = [
#         ('train', 'train_annot', 'train_sorted', 'train_annot_sorted'),
#         ('test', 'test_annot', 'test_sorted', 'test_annot_sorted'),
#         ('valid', 'valid_annot', 'validation_sorted', 'validation_annot_sorted'),
#     ]
#     with ThreadPoolExecutor(max_workers=32) as executor:
#         for image_folder_name, mask_folder_name, output_image_folder_name, output_mask_folder_name in folders:
#             image_folder = os.path.join(root_dir, image_folder_name)
#             mask_folder = os.path.join(root_dir, mask_folder_name)
#             output_image_dir = os.path.join(root_dir, output_image_folder_name)
#             output_mask_dir = os.path.join(root_dir, output_mask_folder_name)
#             # List only .images files in the image folder
#             image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.images')]
#             for image_file in image_files:
#                 base_name = os.path.splitext(image_file)[0]
#                 image_path = os.path.join(image_folder, image_file)
#                 # Corresponding mask file with .png extension
#                 mask_filename = base_name + '_mask.png'
#                 mask_path = os.path.join(mask_folder, mask_filename)
#                 if not os.path.exists(mask_path):
#                     logging.error(f"Mask not found for {image_path}. Supposed to be at {mask_path}")
#                     raise FileNotFoundError()  # Skip if the corresponding mask doesn't exist
#                 executor.submit(process_image, image_path, mask_path, output_image_dir, output_mask_dir)


