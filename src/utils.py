import logging
import os
import shutil
from enum import Enum
from typing import Type, Optional, Any, Union

import h5py
from PIL import Image
import json
import joblib
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
import io
import cv2
import matplotlib.pyplot as plt
from piq import ssim, multi_scale_ssim as ms_ssim
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import torch.nn.functional as F
from segmentation_models_pytorch.utils.metrics import IoU
from tqdm import tqdm

from src.config import setup_logging

setup_logging()


# Decorators
def check_is_image(func: callable):
    def wrapper(image, *args, **kwargs):
        if isinstance(image, np.ndarray):
            if not len(image.shape) == 3:
                raise ValueError(f"Image must have shape (H, W, C) for numpy arrays or (C, H, W) for tensors. Got {image.shape}.")
            if image.min() < 0 or image.max() > 255:
                raise ValueError(f"Image values must be in the range [0, 255]. Got min={image.min()} and max={image.max()}.")
        elif torch.is_tensor(image):
            if image.min().item() < 0.0 or image.max().item() > 1.0:
                raise ValueError(f"Image values must be in the range [0, 1] for tensors. Got min={image.min().item()} and max={image.max().item()}.")
        else:
            raise ValueError(f"Input must be a numpy array or a tensor, not {type(image)}.")
        return func(image, *args, **kwargs)
    return wrapper


def get_centroids(data: np.ndarray, num_clusters: int):
    """
    Get the centroids of the clusters using KMeans. `data`should best be a numpy array.

    :param data: Data to cluster
    :type data: np.ndarray
    :param num_clusters: Number of clusters
    :type num_clusters: int

    :return: Centroids of the clusters, the corresponding labels and the locations of the centroids
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)

    return kmeans.cluster_centers_, kmeans.labels_


def cluster_and_return_labels(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster the given data using KMeans and return the labels.

    :param data: Data to cluster
    :param n_clusters: Number of clusters

    :return: Labels of the clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)


def create_and_plot_synthetic_data(lower_limit: float,
                                   upper_limit: float,
                                   num_samples: int,
                                   plot_type: str = 'scatter'):
    """
    Generates synthetic data and plots it.

    :param lower_limit: Lower limit for the data
    :type lower_limit: float
    :param upper_limit: Upper limit for the data
    :type upper_limit: float
    :param num_samples: Number of samples to generate
    :type num_samples: int
    :param plot_type: Type of plot ('scatter' or 'linear')
    :type plot_type: str

    :return: x and y values
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    x = np.linspace(lower_limit, upper_limit, num_samples)
    y = np.random.uniform(lower_limit, upper_limit, num_samples)

    plt.figure(figsize=(10, 6))
    if plot_type == 'scatter':
        plt.scatter(x, y)
    elif plot_type == 'linear':
        plt.plot(x, y)
    else:
        raise ValueError("plot_type must be either 'scatter' or 'linear'")

    plt.title(f"Synthetic Data Plot ({plot_type})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    return x, y


def rgb_to_mask(rgb_mask: torch.Tensor, class_colors: dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Converts RGB mask image to class index mask image.
    **Note**: broadcast the mask to shape (3xHxW) before passing it to this method.

    :param rgb_mask: RGB mask image tensor with shape (3, H, W)
    :param class_colors: Dictionary containing class enum objects as key and normalized RGB color as value

    :return: Class index mask image with shape (H, W)
    """
    if not rgb_mask.shape[0] == 3:
        raise ValueError(
            f"RGB mask image has to have shape (3, H, W). Got shape: {rgb_mask.shape} Use `torch.permute` to change the shape.")

    mask = torch.zeros((rgb_mask.shape[-2], rgb_mask.shape[-1]), dtype=torch.float32)
    for cls, color in class_colors.items():
        mask[torch.all(rgb_mask == color.view(3, 1, 1), axis=0)] = cls.value if isinstance(cls, Enum) else cls
    return mask.to(torch.int64)


def mask_to_rgb(class_mask: torch.Tensor , class_colors: dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Converts class index mask image to RGB mask image.

    :param class_mask: Class index mask image tensor with shape (H, W)
    :param class_colors: Dictionary containing class enum objects as key and normalized RGB color as value

    :return: RGB mask image tensor with shape (3, H, W)
    """
    if len(class_mask.shape) != 2:
        raise ValueError(f"Class mask image has to have shape (H, W). Got shape: {class_mask.shape}")

    rgb_mask = torch.zeros((3, class_mask.shape[0], class_mask.shape[1]), dtype=torch.float32)
    for cls, color in class_colors.items():
        rgb_mask[:, class_mask == (cls.value if isinstance(cls, Enum) else cls)] = color.view(3, 1)
    return rgb_mask


def load_json(file_path: str) -> dict:
    """
    Load data from a JSON file.

    :param file_path: Path to the JSON file

    :return: Dictionary containing data from the JSON file
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def append_json_list(file_path: str, keyval: dict[str, list[float]]) -> None:
    """
    Appends new elements to the existing lists in a JSON file.

    :param file_path: Path to the JSON file
    :param keyval: Dictionary containing key-value pairs

    :raises NotImplementedError: If the key is not found in the JSON file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    for key, val in keyval.items():
        try:
            if key in data:
                data[key].extend(val)
            else:
                data[key] = val
        except KeyError as e:
            raise NotImplementedError(f"Key {key} not found in the JSON file.") from e

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def save_json(file_path: str, data: dict) -> None:
    """
    Save the given data to a JSON file.

    :param file_path: Path to the JSON file
    :param data: Dictionary containing data to save
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def save_to_hdf5(file_path: str,
                 dataset_dict: dict[str, Any]) -> None:
    """
    Save data to an HDF5 file using concise match-case type handling.

    :param file_path: Path to the HDF5 file
    :param dataset_dict: Dictionary containing data to save

    :raises TypeError: If the data type is not supported
    """
    with h5py.File(file_path, 'w') as f:
        for dataset_name, data in dataset_dict.items():
            match data:
                case torch.Tensor():
                    # Convert Torch tensor to NumPy array
                    data = data.numpy()
                    f.create_dataset(dataset_name, data=data)

                case np.ndarray():
                    # Handle strings in NumPy arrays differently
                    if data.dtype.kind in {'U', 'S'}:  # Unicode or bytes
                        dt = h5py.string_dtype(encoding='utf-8')
                        f.create_dataset(dataset_name, data=data.astype(dt))
                    else:
                        f.create_dataset(dataset_name, data=data)

                case list():
                    # Convert lists to NumPy arrays if possible
                    try:
                        np_data = np.array(data)
                        if np_data.dtype.kind in {'U', 'S'}:
                            dt = h5py.string_dtype(encoding='utf-8')
                            np_data = np_data.astype(dt)
                        f.create_dataset(dataset_name, data=np_data)
                    except ValueError as e:
                        raise ValueError(f"Cannot convert list to NumPy array for dataset '{dataset_name}': {e}")

                case str() | bytes():
                    # Handle single strings or bytes
                    dt = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset(dataset_name, data=np.array([data], dtype=dt))

                case _:
                    raise TypeError(f"Unsupported data type for dataset '{dataset_name}': {type(data)}")

def load_hdf5(file_path: str) -> dict[str, np.ndarray]:
    """
    Load data from an HDF5 file.

    :param file_path: Path to the HDF5 fileuse

    :return: Dictionary containing data from the HDF5 file
    """
    with h5py.File(file_path, 'r') as file:
        data = {key: val[:] for key, val in file.items()}
    return data


def mean_below_diagonal(matrix: np.ndarray) -> float:
    """
    Calculate the mean of elements below the diagonal of a symmetric matrix.

    :param matrix: Symmetric numpy array with 1s on the diagonal.
    :return: Mean of the elements below the diagonal.
    """
    below_diag_elements = matrix[np.tril_indices_from(matrix, k=-1)]
    mean_value = below_diag_elements.mean()
    return mean_value


def soft_dice_score(output: torch.Tensor,
                    target: torch.Tensor,
                    smooth: float = 0.0,
                    eps: float = 1e-7,
                    dims=None) -> torch.Tensor:
    """
    Compute the Soft Dice Score for the given output and target.

    :param output: Model output. Shape: (N, C, HxW)
    :param target: Target mask. Shape: (N, C, HxW)
    :param smooth: label smoothing value
    :param eps: epsilon value to avoid division by zero
    :param dims: dimensions to reduce. Default is None

    :return: soft dice score
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def multi_class_dice_score(pred_mask: torch.Tensor,
                           true_mask: torch.Tensor,
                           num_classes: int,
                           ignore_channels: list[int] = None) -> torch.Tensor:
    """
    # TODO: this method is deprecated. Use 'soft_dice_score' instead.
    Compute Dice Similarity Coefficient for the predicted mask and the true mask.

    :param pred_mask: predicted mask with shape (H, W)
    :param true_mask: true mask with shape (H, W)
    :param num_classes: number of classes
    :param ignore_channels: list of classes to ignore while calculating the Dice score (pass 0 to ignore the background class)

    :return: Dice score

    :raises ValueError: If the shape of the predicted mask and the true mask is not the same
    :raises ValueError: If the predicted mask and the true mask are not 2D tensors
    """
    if not pred_mask.shape == true_mask.shape:
        raise ValueError("The shape of the predicted mask and the true mask should be the same.")
    if not len(pred_mask.shape) == 2 or not len(true_mask.shape) == 2:
        raise ValueError(
            f"The predicted mask and the true mask should be 2D tensors, got {pred_mask.shape} for prediction and {true_mask.shape} for ground truth.")

    # Convert pred and true masks to one-hot encoded format
    pred_mask_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(2, 0, 1).float()
    true_mask_one_hot = F.one_hot(true_mask, num_classes=num_classes).permute(2, 0, 1).float()
    # Initialize a list to store Dice scores for each class
    dice_scores = []

    # Calculate Dice score for each class, ignoring the specified ignore_channels
    for class_index in range(num_classes):
        if ignore_channels:
            if class_index in ignore_channels:
                logging.info("Ignoring channel:", class_index)
                continue

        # Get the predicted and true masks for the current class
        pred_class, true_class = pred_mask_one_hot[class_index], true_mask_one_hot[class_index]

        # Calculate intersection and union
        intersection = torch.sum(pred_class * true_class)
        pred_sum, true_sum = torch.sum(pred_class), torch.sum(true_class)

        # Compute Dice coefficient for this class
        dice_score = (2.0 * intersection) / (pred_sum + true_sum + 1e-8)  # Small epsilon to avoid division by zero
        dice_scores.append(dice_score)

    return torch.mean(torch.tensor(dice_scores))


def multiclass_iou(pred_mask: torch.Tensor,
                   true_mask: torch.Tensor,
                   num_classes: int,
                   ignore_channels: list = None) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) for the predicted mask and the true mask.

    :param pred_mask: predicted mask with shape (H, W)
    :param true_mask: true mask with shape (H, W)
    :param num_classes: number of classes
    :param ignore_channels: list of channels to ignore (pass 0 to ignore background)

    :return: IoU score

    :raises ValueError: If the shape of the predicted mask and the true mask is not the same
    :raises ValueError: If the predicted mask and the true mask are not 2D tensors
    """
    if not pred_mask.shape == true_mask.shape:
        raise ValueError("The shape of the predicted mask and the true mask should be the same.")
    if not len(pred_mask.shape) == 2 or not len(true_mask.shape) == 2:
        raise ValueError(f"The predicted mask and the true mask should be 2D tensors, got {pred_mask.shape} for prediction and {true_mask.shape} for ground truth.")

    pred_mask_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()
    true_mask_one_hot = F.one_hot(true_mask, num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()

    return IoU(eps=1e-6, threshold=None, ignore_channels=ignore_channels)(pred_mask_one_hot, true_mask_one_hot)


def get_enum_member(cls_of_interest: str, enum_class: Type[Enum]) -> Optional[Enum]:
    """
    Retrieve an enum member by its name (case-insensitive) from a given enum class.

    :param cls_of_interest: The name of the enum member to look up.
    :param enum_class: The enum class to search within.

    :returns: The corresponding enum member if found, otherwise None.
    """
    cls_name = cls_of_interest.upper()
    return enum_class.__members__.get(cls_name)

def calc_ssim(image_1: torch.Tensor, image_2: torch.Tensor) -> list[torch.Tensor]:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    :param image_1: First image
    :param image_2: Second image

    :return: SSIM value
    """
    if not isinstance(image_1, torch.Tensor) or not isinstance(image_2, torch.Tensor):
        raise ValueError(f"Both images must be of type torch.Tensor, but got {type(image_1)} and {type(image_2)} instead.")
    return ssim(image_1.unsqueeze(0), image_2.unsqueeze(0), data_range=1.0)


def calc_ms_ssim(image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two images.
    :param image_1: First image
    :param image_2: Second image

    :return: MS-SSIM value
    """
    if not isinstance(image_1, torch.Tensor) or not isinstance(image_2, torch.Tensor):
        raise ValueError(f"Both images must be of type torch.Tensor, but got {type(image_1)} and {type(image_2)} instead.")
    return ms_ssim(image_1.unsqueeze(0), image_2.unsqueeze(0), data_range=1.0)


def cluster_images_and_save(
        image_paths: list[str],
        features: np.ndarray | torch.Tensor,
        n_clusters: int,
        output_dir: str,
        generate_heatmap: bool = True,
        heatmap_title: str = "Heatmap",
        rename_images: bool = True,
        verbose: bool = True) -> None:
    """
    Cluster images based on provided features and save them into folders.

    :param image_paths: List of image paths.
    :param features: Feature matrix of shape (num_images, feature_dimension).
    :param n_clusters: Number of clusters to form.
    :param output_dir: Directory to save clustered images and heatmaps.
    :param generate_heatmap: Whether to generate heatmaps for each cluster.
    :param heatmap_title: Title for the heatmaps.
    :param rename_images: Whether to rename images in the cluster folders. The schema is simply `image_{index}.jpg`.
    :param verbose: Whether to print progress messages.
    """
    if verbose:
        logging.info(f"Clustering {len(image_paths)} images into {n_clusters} clusters...")
    labels = cluster_and_return_labels(features, n_clusters=n_clusters)

    for cluster_num in tqdm(range(n_clusters), desc="Processing clusters"):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_image_paths = [image_paths[idx] for idx in cluster_indices]

        # Create cluster directory
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_num}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Copy and optionally rename images
        cluster_index_list = []
        for idx, image_path in enumerate(cluster_image_paths, start=1):
            if rename_images:
                # Determine file extension
                _, ext = os.path.splitext(image_path)
                image_filename = f"image_{idx}{ext}"
            else:
                image_filename = os.path.basename(image_path)

            dest_path = os.path.join(cluster_dir, image_filename)
            shutil.copy(image_path, dest_path)
            cluster_index_list.append(str(idx))

        if generate_heatmap:
            if len(cluster_indices) > 1:
                cluster_features = features[cluster_indices]
                similarity_matrix = cosine_similarity(cluster_features)
                try:
                    plot_and_save_heatmap(
                        matrix=similarity_matrix,
                        title=f"{heatmap_title} - Cluster {cluster_num} - {mean_below_diagonal(similarity_matrix):.2f} Avr Similarity",
                        x_tick_labels=cluster_index_list,
                        y_tick_labels=cluster_index_list,
                        cbar_kws={"label": "Cosine Similarity"},
                        save_fig_path=os.path.join(cluster_dir, f"heatmap_cluster_{cluster_num}.png"),
                        show=False)
                except ValueError as e:
                    logging.warning(f"Error generating heatmap for cluster {cluster_num}: {e}. Probably the heatmap is too large.")
                save_to_hdf5(file_path=os.path.join(cluster_dir, f"heatmap_cluster_{cluster_num}.h5"),
                              dataset_dict={"heatmap": similarity_matrix,
                               "image_paths": [path.replace('/', '|').replace('\\', '|') for path in cluster_image_paths]})
                save_json(file_path=os.path.join(cluster_dir, f"cluster_{cluster_num}_info.json"),
                          data={"num_images": len(cluster_indices),
                                "indices": cluster_index_list,
                                "average_similarity": mean_below_diagonal(similarity_matrix)})
            else:
                if verbose:
                    logging.info(f"Cluster {cluster_num} contains only one image; skipping heatmap generation.")

    if verbose:
        logging.info(f"Clustering completed. Results saved in {output_dir}")


@check_is_image
def permute_image_channels(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Permute image to shape (C, H, W) if the current shape is (H, W, C).

    :param image: Input image

    :return: Permuted image
    """
    if image.shape[2] == 3:
        return np.transpose(image, (2, 0, 1)) if isinstance(image, np.ndarray) else image.permute(2, 0, 1)

@check_is_image
def get_non_zero_pixel_indices(image: np.ndarray) -> tuple:
    """
    Get the indices of pixels that have at least one non-zero channel.
    Use this method to find coordinates of non-black pixels in an image.

    :param image: Input image
    :type image: np.ndarray
    :return: Indices of non-zero pixels in a tuple
    :rtype: tuple
    """
    return tuple(np.argwhere(np.any(image != 0, axis=-1)))


@check_is_image
def plot_clusters_on_image(image: np.ndarray,
                           data: np.ndarray,
                           labels: np.ndarray,
                           keypoints: list[cv2.KeyPoint]) -> None:
    """
    Plot two images with keypoints next to each other:
    - The left image shows which keypoint belongs to which cluster by using a color code.
    - The second plot display more rich information about the keypoints (e.g. size, orientation, etc.).

    :param image: Image to display as the background
    :type image: bp.ndarray
    :param data: 2D numpy array of data points (n_samples, n_features)
    :type data: np.ndarray
    :param labels: 1D numpy array of cluster labels for each data point
    :type labels: np.ndarray
    :param keypoints: List of cv2.KeyPoint objects
    :type keypoints: list[cv2.KeyPoint]
    """

    # Convert image to rgb format
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define unique cluster labels
    unique_labels = np.unique(labels)

    # Generate a color map with as many colors as unique clusters
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    # Create a figure with extra space on the right for the legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Display the image as the background
    left, right, bottom, top = 0, img.shape[1], image.shape[
        0], 0  # Image is upside down on the y-axis when calling plt.imshow, hence bottom, top are flipped
    ax1.imshow(img, extent=[left, right, bottom, top])

    # Plot each cluster with a different color
    for cluster_idx in unique_labels:
        # Select data points belonging to the current cluster
        cluster_data = data[labels == cluster_idx]

        ax1.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    s=20, c=[colors(cluster_idx)],
                    label=f"Cluster {cluster_idx}", alpha=0.7)

    # # Plot the centroids in a different style
    # ax1.scatter(centroids[:, 0], centroids[:, 1],
    #             s=120, c='black', marker='x',
    #             label='Centroids')

    ax1.set_title("Clusters and Centroids on Image")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

    # Keep axes in the range of the image size
    ax1.set_xlim(0, img.shape[1])
    ax1.set_ylim(img.shape[0], 0)  # Flip the y-axis to match the image orientation

    # Place legend outside the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # If show_centroid_coords is True, display the coordinates of the centroids
    # if show_centroid_coords:
    #     for i, centroid in enumerate(centroids):
    #         ax1.text(centroid[0], centroid[1], f"Centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})", fontsize=6,
    #                  color='black')
    #
    # # If show_centroid_labels is True, display the labels of the centroids
    # if show_centroid_labels:
    #     for i, centroid in enumerate(centroids):
    #         ax1.text(centroid[0], centroid[1], f"Cluster {i}", fontsize=6, color='black')

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax2.imshow(img_with_keypoints)
    ax2.set_title("Keypoints on Image")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_xlim(0, img.shape[1])
    ax2.set_ylim(img.shape[0], 0)  # Flip the y-axis to match the image orientation

    plt.subplots_adjust(wspace=0.5)
    plt.grid(False)
    plt.show()


def plot_and_save_heatmap(matrix: Union[list, np.ndarray, torch.Tensor],
                          x_tick_labels: list[str]=None,
                          y_tick_labels: list[str]=None,
                          cbar_kws: dict[str, str]=None,
                          title: str="Heatmap",
                          x_label: str="X Axis",
                          y_label: str="Y Axis",
                          show: bool=True,
                          save_fig_path: str=None) -> None:
    """
    Plot a heatmap using the specified matrix.

    :param matrix: matrix
    :param x_tick_labels: x-axis tick labels
    :param y_tick_labels: y-axis tick labels
    :param cbar_kws: colorbar keyword arguments
    :param title: title of the plot
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param show: whether to display the plot
    :param save_fig_path: Path to save the figure
    **kwargs: Additional keyword arguments (currently available: title, xlabel, ylabel)
    """
    plt.figure(figsize=(len(matrix) * 0.7, len(matrix) * 0.7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=x_tick_labels if x_tick_labels else list(range(matrix.shape[1])),
                yticklabels=y_tick_labels if y_tick_labels else list(range(matrix.shape[0])),
                cbar_kws=cbar_kws if cbar_kws else {"label": "value"})
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_fig_path:
        plt.savefig(save_fig_path)
    if show:
        plt.show()
    plt.close()


def is_subset(list1: list, list2: list) -> bool:
    """
    Check if list1 is a subset of list2.

    :param list1: First list to check (potential subset)
    :param list2: Second list (or tuple) to check against (potential superset)

    :returns: True if list1 is a subset of list2, False otherwise
    """
    if len(list1) > len(list2):
        raise ValueError("List1 must be have smaller or equal length than list2")
    return set(list1).issubset(list2)


def convert_to_integers(list_of_tuples: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """
    Convert all elements in a list of tuples to integers.

    :param list_of_tuples: List of tuples with float values

    :return: List of tuples with integer values
    """
    return [(int(x), int(y)) for x, y in list_of_tuples]


def standardize_data(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Standardize the given data using the formula: (x - mean) / std.

    :param data: Input data
    :param axis: Axis along which to standardize the data (for row-wise standardization, use axis=0. For column-wise standardization, use axis=1)

    :return: Standardized data
    """
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=0)


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.

    :param model: Model to save
    :param file_path: Path to save the trained model
    """
    with open(file_path, 'wb') as file:
        joblib.dump(model, file)


def load_model(file_path: str) -> object:
    """
    Load a pre-trained model from a file.

    :param file_path: Path from which to load the trained model

    :return: Trained model
    """
    with open(file_path, 'rb') as file:
        return joblib.load(file)


@check_is_image
def average(matrix: np.ndarray | torch.Tensor) -> float:
    """
    Compute the average of the given matrix.

    :param matrix: Input matrix

    :return: Average value
    """
    return np.mean(matrix) if isinstance(matrix, np.ndarray) else torch.mean(matrix).item()


@check_is_image
def gaussian_blur(image: np.ndarray | torch.Tensor, kernel_size: int=3, sigma: float=1.0) -> np.ndarray | torch.Tensor:
    """
    Apply Gaussian blurring to the given image.

    :param image: Input image
    :param kernel_size: Size of the kernel
    :param sigma: Standard deviation of the kernel

    :return: Blurred image
    """
    min_kernel_size = 2 * int(3 * sigma) + 1
    max_kernel_size = 2 * int(5 * sigma) + 1
    if not min_kernel_size <= kernel_size <= max_kernel_size:
        raise ValueError(f"Kernel radius must be between 2 * 3-5 times the standard deviation plus one. " 
                         f"In this case, it should be between {min_kernel_size} and {max_kernel_size} "
                         f"Got kernel size: {kernel_size}")
    if isinstance(image, np.ndarray):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif torch.is_tensor(image):
        return TF.gaussian_blur(image, [kernel_size, kernel_size], [sigma, sigma]).clamp(0.0, 1.0)


@check_is_image
def compress_image(image: np.ndarray | torch.Tensor, quality: int) -> np.ndarray | torch.Tensor:
    """
    Compress the image using JPEG compression at the specified quality.

    :param image: Input image
    :param quality: Quality for JPEG compression (0 to 100).
    :return: Compressed image
    """
    # Ensure quality is within the valid range
    quality = max(0, min(quality, 100))
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        img = Image.fromarray(image.astype(np.uint8))
        # Compress image using BytesIO
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        # Read image back from buffer
        compressed_img = Image.open(buffer)
        compressed_image = np.array(compressed_img)
        return compressed_image
    elif torch.is_tensor(image):
        if image.shape[0] not in (1, 3):
            image = image.permute(2, 0, 1)

        # Convert to unit8 tensor
        image_unit8 = (image * 255).clamp(0, 255).to(torch.uint8)
        # Conpress image
        encoded_jpeg = torchvision.io.encode_jpeg(image_unit8, quality=quality)
        # Decode image
        compressed_image = torchvision.io.decode_jpeg(encoded_jpeg).float() / 255
        # Ensure the output has the same device and dtype as input
        compressed_image = compressed_image.to(image.dtype).to(image.device).clamp(0.0, 1.0)
        return compressed_image


@check_is_image
def thresholding(image, threshold_value=None, max_value=255, threshold_types: tuple = (cv2.THRESH_BINARY,)):
    """
    Currently only works for gray images.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, np.sum(threshold_types))
    logging.debug(f"Threshold value used: {threshold_value}")
    return thresholded_image

@check_is_image
def resize(image, dimensions, interpolation=cv2.INTER_LINEAR):
    """
    Resizes the given image to the given dimensions. If a single integer is passed,
    both the width and height will be resized to that integer. If a tuple is passed,
    then it works like the normal cv2.resize function.
    """
    if isinstance(dimensions, int):
        dimensions = (dimensions, dimensions)
    return cv2.resize(image, dimensions, interpolation=interpolation)


@check_is_image
def sharpen(image, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
    """
    Sharpens the given image using the given kernel.
    """
    return cv2.filter2D(image, -1, kernel)


@check_is_image
def plot_image(image: np.ndarray | torch.Tensor, title: str = None) -> None:
    """
    Plot the image with its file path and label.
    If image shape of (3, width, height) is passed, it is converted to (width, height, 3) before plotting.
    """
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


@check_is_image
def sift(image) -> [cv2.KeyPoint, np.ndarray]:
    """
    Extracts SIFT features from the given image.

    :param image: Input image
    :type image: np.ndarray

    :return: keypoints as a list of cv2.KeyPoint objects and descriptors as a numpy array
    :rtype: tuple
    """
    sift = cv2.SIFT.create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


@check_is_image
def root_sift(image: np.ndarray,
              epsilon: float = 1e-7) -> tuple[cv2.KeyPoint, np.ndarray]:
    """
    Extracts RootSIFT features from the given image. The only difference to SIFT is that
    L1 normalization and square root are applied to the descriptors before any further processing.

    :param image: Input image
    :param epsilon: Small value to avoid division by zero
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + epsilon)
    descriptors = np.sqrt(descriptors)
    return keypoints, descriptors


@check_is_image
def surf(image: np.ndarray) -> tuple[cv2.KeyPoint, np.ndarray]:
    """
    Extracts SURF features from the given image.
    """
    surf = cv2.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors


@check_is_image
def difference_of_gaussian(image: np.ndarray,
                           num_intervals: int,
                           num_octaves: int = 1,
                           sigma: float = 1.6,
                           plot: bool = False) -> list:
    """
    Calculates DoG for the given image.

    :param image: Input image
    :type image: np.ndarray
    :param num_intervals: Number of intervals (normally written as `s`) in each octave
    :type num_intervals: int
    :param num_octaves: Number of octaves
    :type num_octaves: int
    :return: List of octave images (the difference of gaussian images within each octave)
    :rtype: list
    """
    k = 2 ** (1.0 / num_intervals)  # Scale factor
    octave_images = []
    octave_range = num_intervals + 3

    # Generate blurred images
    for octave in range(num_octaves):
        gaussian_images = []
        current_sigma = sigma
        logging.info(f"""
        Calculating DoG for octave {octave} with {num_intervals} intervals and sigma={sigma}:
        
        =====================================================================================\n
        """)

        for _ in range(octave_range):
            gaussian_images.append(gaussian_blur(image, sigma=current_sigma))
            logging.debug(f"Sigma value used: {current_sigma}")
            current_sigma *= k

        # Calculate DoG and append to the octave images
        for i in range(1, len(gaussian_images)):
            dog = gaussian_images[i] - gaussian_images[i - 1]
            octave_images.append(dog)

        # Downsample the image by factor of 2 for the next octave
        logging.debug(f"Current image shape: {image.shape}")
        image = resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    logging.debug("Total number of octave images: %s", len(octave_images))
    if plot:
        plt.figure(figsize=(25, 10))
        for i in range(num_octaves):
            for j in range(num_intervals + 2):
                plt.subplot(num_octaves, num_intervals + 2, i * (num_intervals + 2) + j + 1)
                plt.title(f"Octave: {i}, Interval: {j}")
                plt.imshow(cv2.cvtColor(octave_images[i * (num_intervals + 2) + j], cv2.COLOR_BGR2RGB))

        plt.suptitle(f"""
            "Difference of Gaussian calculation with initial_sigma=1.6\n"
            "Number of intervals: {num_intervals}, Number of octaves: {num_octaves}",
            fontsize=20
            """)
        plt.show()
    return octave_images


@check_is_image
def denoise_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Denoises the input binary mask image by removing components smaller than the specified minimum size.

    :param mask: Input binary mask (HxW)
    :param min_size: Minimum pixel size for components to retain

    :returns: Processed mask with only components of size >= min_size, retaining original class values.
    """
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    for class_value in np.unique(mask):
        logging.debug("List of classes before being filtered: %s", class_value)
        if class_value == 0:  # Skip the background class
            continue

        class_mask = np.where(mask == class_value, 255, 0).astype(
            np.uint8)  # Classes of question are set to 255. The rest is 0.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=4)
        for i in range(1, num_labels):
            component_size = stats[i, cv2.CC_STAT_AREA]
            if component_size >= min_size:
                filtered_mask[labels == i] = class_value

    logging.debug("List of classes after being filtered: %s", np.unique(filtered_mask))
    return filtered_mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets import BaseDataset

    flower_data = BaseDataset('data/raw/train')
    image = flower_data[20][0]
    octave_images = difference_of_gaussian(image,
                                           num_intervals=5,
                                           num_octaves=2,
                                           plot=True)
