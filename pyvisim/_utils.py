import io
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Type, Optional, Any, Union, Sequence, Literal

import cv2
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity as cs
from tqdm import tqdm

from ._config import setup_logging
from ._errors import InvalidImageError

setup_logging()


# Decorators
def is_numpy_image(image: np.ndarray, pos: int) -> None:
    """
    Check if a numpy array is a valid image.

    :param image: The numpy array to validate.
    :param pos: Position of the image in the argument list (for error messages).
    :raises InvalidImageError: If the image is invalid.
    """
    if len(image.shape) == 2:
        if not np.all(image == image.astype(np.int64)):
            raise InvalidImageError(
                f"Mask values must be integers. Got min={image.min()} and max={image.max()}."
            )
    else:
        if image.shape[2] != 3:
            raise InvalidImageError(f"NumPy 3D images must have shape (H, W, 3). Got {image.shape}.")
        if image.min() < 0 or image.max() > 255:
            raise InvalidImageError(
                f"Image values must be in the range [0, 255]. Got min={image.min()} and max={image.max()} for position {pos}."
            )

def is_torch_image(image: torch.Tensor, pos: int, tol: float) -> None:
    """
    Check if a PyTorch tensor is a valid image.

    :param image: The PyTorch tensor to validate.
    :param pos: Position of the image in the argument list (for error messages).
    :param tol: Tolerance for float comparison.
    :raises InvalidImageError: If the image is invalid.
    """
    if len(image.shape) == 2:
        if not torch.all(image == image.to(torch.int)):
            raise InvalidImageError(
                f"Mask values must be integers. Got min={image.min().item()} and max={image.max().item()} for position {pos}."
            )
    else:
        if image.shape[0] != 3:
            raise InvalidImageError(f"Torch 3D images must have shape (3, H, W). Got {image.shape}.")
        if image.min().item() < 0.0 - tol or image.max().item() > 1.0 + tol:
            raise InvalidImageError(
                f"Image values must be in the range [0, 1]. Got min={image.min().item()} and max={image.max().item()} for position {pos}."
            )

def check_is_image(arg_positions: tuple = None, kwarg_positions: tuple = None, tol=1e-5):
    """
    Decorator to check if one or more arguments are valid images. Default is the first positional
    argument.

    **Note**: both 'arg_positions' and 'kwarg_positions' are zero-based!

    :param arg_positions: A tuple of positions (integers) in *args
                          that correspond to the position of the images / masks.
    :param kwarg_positions: A tuple of positions (integers) in **kwargs
                            that correspond to the position of the images / masks.
    :param tol: Tolerance for float comparison for Torch images.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            def generator(*gen_args, **gen_kwargs):
                if arg_positions:
                    for pos in arg_positions:
                        yield gen_args[pos]
                if kwarg_positions:
                    kw_vals = (val for i, val in enumerate(gen_kwargs.values()) if i in kwarg_positions)
                    for kw_val in kw_vals:
                        yield kw_val
                if not arg_positions and not kwarg_positions:
                    yield gen_args[0]

            for pos, image in enumerate(generator(*args, **kwargs)):
                if not hasattr(image, 'shape'):
                    raise InvalidImageError(
                        f"Argument at position {pos} of type {type(image)} does not have attribute 'shape'. So it is neither a numpy array nor a torch tensor."
                    )
                if not (2 <= len(image.shape) <= 3):
                    raise InvalidImageError(
                        f"Image must be 2D or 3D. Got shape {image.shape} for position {pos}."
                    )

                if isinstance(image, np.ndarray):
                    is_numpy_image(image, pos)
                elif torch.is_tensor(image):
                    is_torch_image(image, pos, tol)
                else:
                    raise InvalidImageError(
                        f"Input must be a numpy array or a torch tensor, not {type(image)}."
                    )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def cluster_and_return_labels(data: np.ndarray,
                              method: Literal['kmeans', 'dbscan', 'spectral'] = 'kmeans',
                              n_clusters: Optional[int] = None,
                              **kwargs) -> np.ndarray:
    """
    Clusters 'data' using the specified method.

    :param data: A 2D NumPy array of shape (N, D)
    :param method: 'kmeans', 'dbscan', or 'spectral'
    :param n_clusters: Number of clusters (if applicable)
    :param kwargs: Additional arguments to pass to the clustering constructor
    :return: 1D NumPy array of cluster labels (shape: (N,))
    """
    if method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans.")
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = model.fit_predict(data)
        return labels

    elif method == 'dbscan':
        # DBSCAN doesn't need n_clusters (but can accept eps, min_samples)
        model = DBSCAN(**kwargs)
        labels = model.fit_predict(data)
        return labels

    elif method == 'spectral':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Spectral Clustering.")
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, **kwargs)
        labels = model.fit_predict(data)
        return labels

    else:
        raise ValueError(f"Unknown method: {method}")


@check_is_image()
def plot_image(image: np.ndarray | torch.Tensor, title: str = "Image") -> None:
    """Simply olots the image."""
    plt.figure(figsize=(10, 10))
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()


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
        def save_to_hdf5_helper(dataset_dict: dict[str, Any], f: h5py.File):
            for dataset_name, data in dataset_dict.items():
                match data:
                    case int() | float():
                        f.create_dataset(dataset_name, data=data)

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

                    case dict():
                        # Recursively save nested dictionaries
                        group = f.create_group(dataset_name)
                        save_to_hdf5_helper(data, group)
                    case _:
                        raise TypeError(f"Unsupported data type for dataset '{dataset_name}': {type(data)}")

        save_to_hdf5_helper(dataset_dict, f)

def load_hdf5(file_path: str) -> dict[str, any]:
    """
    Load data from an HDF5 file into a dictionary with proper type handling.

    :param file_path: Path to the HDF5 file.
    :return: Dictionary with dataset names as keys and data with proper types.
    """
    with h5py.File(file_path, 'r') as file:
        def load_hdf5_helper(file: h5py.File) -> dict[str, any]:
            data = {}
            for key, val in file.items():
                if isinstance(val, h5py.Group):
                    # Recursively load groups as nested dictionaries
                    data[key] = load_hdf5_helper(val)
                elif isinstance(val, h5py.Dataset):
                    # Check the dtype of the dataset
                    if val.dtype.kind in {'U', 'S'}:  # String or bytes
                        data[key] = val.asstr()[...]
                    elif val.shape == ():  # Scalar
                        data[key] = val[()]
                    else:
                        data[key] = val[...]
            # Add attributes as additional keys
            for attr_key, attr_val in file.attrs.items():
                data[attr_key] = attr_val
            return data

        return load_hdf5_helper(file)


def mean_below_diagonal(matrix: np.ndarray) -> float:
    """
    Calculate the mean of elements below the diagonal of a symmetric matrix.

    :param matrix: Symmetric numpy array with 1s on the diagonal.
    :return: Mean of the elements below the diagonal.
    """
    below_diag_elements = matrix[np.tril_indices_from(matrix, k=-1)]
    mean_value = below_diag_elements.mean()
    return float(mean_value)


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


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between two matrices.

    :param x: First matrix
    :param y: Second matrix

    :return: Cosine similarity matrix
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    x = x.reshape(1, -1) if len(x.shape) == 1 else x
    y = y.reshape(1, -1) if len(y.shape) == 1 else y
    if x.shape[-1] <= 1 or y.shape[-1] <= 1:
        raise ValueError(f"Cosine similarity requires at least 2 features. Got {x.shape[-1]} features for x and {y.shape[-1]} features for y.")

    return cs(x, y)


def cluster_images_and_generate_statistics(
    features: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    method: str = 'kmeans',
    **kwargs
) -> dict[str, float]:
    """
    Clusters the given features and computes ARI, NMI.

    :param features: (N, D) array of feature vectors
    :param true_labels: (N,) array of ground truth class labels
    :param n_clusters: Number of clusters to find
    :param method: 'kmeans', 'dbscan', or 'spectral'
    :param kwargs: Additional parameters for the clustering method
    :return: Dictionary of statistics {'ari': ..., 'nmi': ...}
    """
    cluster_labels = cluster_and_return_labels(
        data=features,
        method=method,
        n_clusters=n_clusters if method != 'dbscan' else None,
        **kwargs
    )

    return {
        "ri": rand_score(true_labels, cluster_labels),
        "ari": adjusted_rand_score(true_labels, cluster_labels),
        "nmi": adjusted_mutual_info_score(true_labels, cluster_labels)
    }


def plot_and_save_heatmap(matrix: Union[list, np.ndarray, torch.Tensor],
                          figsize: tuple[int, int]=None,
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
    :param figsize: figure size
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
    figsize = (len(matrix) * 0.7, len(matrix) * 0.7) if figsize is None else figsize
    plt.figure(figsize=figsize)
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

def plot_and_save_barplot(data: dict[str, list[float]],
                            bar_labels: list[str],
                            title: str = "Barplot",
                            xlabel: str = "X-axis",
                            ylabel: str = "Y-axis",
                            save_path: str = None,
                            show: bool = True) -> None:
    """
    Plot and save a barplot.
    :param data: Dictionary containing data to plot.
    :param bar_labels: Labels that will be displayed in the legend.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the plot image. If None, plot is not saved.
    :param show: Whether to display the plot.
    """
    x_labels = list(data.keys())
    values = list(data.values())
    num_groups = len(values[0])

    if not all(len(v) == num_groups for v in values):
        raise ValueError("All lists in data must have the same length as the number of bar labels.")

    x = np.arange(len(x_labels))  # the label locations
    width = 0.8 / num_groups      # width of each bar

    plt.figure(figsize=(10, 6))

    for i in range(num_groups):
        heights = [v[i] for v in values]
        plt.bar(x + i * width, heights, width, label=bar_labels[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x + width * (num_groups - 1) / 2, x_labels)  # Center the tick labels
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def plot_and_save_lineplot(y: np.ndarray,
                           x: np.ndarray = None,
                           y_lim: tuple = None,
                           x_lim: tuple = None,
                           save_path: str = None,
                           sort_y: bool = False,
                           title: str = "Lineplot",
                           xlabel: str = "x-axis",
                           ylabel: str = "y-axis") -> None:
    """
    Plot and save a lineplot, limiting x-ticks to at most 20 evenly distributed values.

    :param y: Array of y-values.
    :param x: Array of x-values. If None, indices of y will be used.
    :param y_lim: Tuple for y-axis limits (min, max). Optional.
    :param x_lim: Tuple for x-axis limits (min, max). Optional.
    :param save_path: Path to save the plot image. If None, plot is not saved.
    :param sort_y: Whether to sort y-values before plotting.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :returns: None
    """
    if x is None:
        x = np.arange(len(y))

    if sort_y:
        y = np.sort(y)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if len(x) > 20:
        tick_indices = np.linspace(0, len(x) - 1, 20, dtype=int)
        tick_labels = [x[i] for i in tick_indices]
        plt.xticks(tick_indices, tick_labels, rotation=90)
    if y_lim:
        plt.ylim(y_lim)
    if x_lim:
        plt.xlim(x_lim)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_and_save_histogram(data: np.ndarray,
                            num_bins: int = 10,
                            title: str = "Histogram",
                            x_label: str = "Value",
                            y_label: str = "Frequency",
                            save_path: str = None,
                            x_lim: tuple = (0, 1),
                            show: bool = True) -> None:
    """
    Plot and save a histogram, allowing the user to choose the number of bins.

    :param data: Array of data to plot.
    :param num_bins: Number of bins for the histogram.
    :param title: Title of the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param save_path: Path to save the plot image. If None, plot is not saved.
    :param x_lim: Tuple for x-axis limits (min, max). Optional.
    :param show: Whether to display the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if x_lim:
        plt.xlim(*x_lim)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def fit_regression_line(x: np.ndarray, y: np.ndarray, poly_degree: int) -> object:
    """
    Fit a polynomial regression line to the given data.

    :param x: Independent variable values (numpy array).
    :param y: Dependent variable values (numpy array).
    :param poly_degree: Degree of the polynomial regression.
    :return: RegressionResult containing predictions, coefficients, and intercept. Attributes: predictions, coefficients, intercept.
    """
    @dataclass
    class RegressionResult:
        predictions: np.ndarray
        coefficients: np.ndarray
        intercept: float
        mse: float

    poly_features = np.vander(x, N=poly_degree + 1, increasing=True)
    reg = LinearRegression().fit(poly_features, y)
    predictions = reg.predict(poly_features)
    mse = mean_squared_error(y, predictions)
    return RegressionResult(predictions, reg.coef_, reg.intercept_, mse)


def get_statistics(x: np.ndarray, y: np.ndarray) -> object:
    """
    Calculate statistics such as Pearson and Spearman correlations, standard deviation,
    mean, median, and number of data points for the given data.

    :param x: Independent variable values (numpy array).
    :param y: Dependent variable values (numpy array).
    :return: Statistics containing computed statistical encoders. Attributes: pearson, spearman, std, mean, median, n_points.
    """
    @dataclass
    class Statistics:
        pearson: float
        spearman: float
        std: float
        mean: float
        median: float
        n_points: int

    pearson, _ = pearsonr(x, y)
    spearman, _ = spearmanr(x, y)
    std, mean, median, n_points = np.std(y), np.mean(y), np.median(y), len(y)
    return Statistics(pearson, spearman, std, mean, median, n_points)


def plot_boxplot_with_regression(x: np.ndarray,
                                 y: np.ndarray,
                                 poly_degree: int = 1,
                                 x_lim: tuple = (0, 1),
                                 y_lim: tuple = (0, 1),
                                 num_bins: int = 20,
                                 title: str = "Boxplot with Regression",
                                 x_label: str = "IoU Difference",
                                 y_label: str = "Similarity Score",
                                 save_fig_path: str = None,
                                 plot_bin_regression: bool = False,
                                 verbose: bool = False,
                                 return_results: bool = False):
    """
    Plot a boxplot of y values binned by x, and add a polynomial regression line over the scatter.

    :param x: Array of x values.
    :param y: Array of y values.
    :param poly_degree: The degree of the polynomial for regression.
    :param x_lim: Tuple of (min, max) x-axis limits.
    :param y_lim: Tuple of (min, max) y-axis limits.
    :param num_bins: Number of bins for the boxplot.
    :param title: Plot title.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param save_fig_path: Optional path to save the plot.
    :param plot_bin_regression: If True, plots regression lines within each bin and prints coefficients.
    :param verbose: If True, prints additional information.
    :param return_results: If True, returns the overall statistics and per-bin statistics.
    """

    def bin_data(x: np.ndarray, y: np.ndarray, lower: float, upper: float, num_bins: int):
        bins = np.linspace(lower, upper, num_bins + 1)
        bin_indices = np.digitize(x, bins) - 1
        binned_y = [[] for _ in range(num_bins)]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for xi, yi, bi in zip(x, y, bin_indices):
            if 0 <= bi < num_bins:
                binned_y[bi].append(yi)
        return bin_centers, binned_y

    lower, upper = x_lim
    bin_centers, binned_y = bin_data(x, y, lower, upper, num_bins)

    # Replace empty bins with NaN to stretch the boxplots evenly
    binned_y = [b if b else [np.nan] for b in binned_y]

    plt.figure(figsize=(12, 8))
    plt.boxplot(binned_y, positions=bin_centers, widths=(upper - lower) / (num_bins * 2), patch_artist=True)

    # Apply regression if valid data exists
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid]
    y_valid = y[valid]

    if not len(x_valid) > 1:  # Ensure enough data points are available
        raise ValueError("Less than two data points are valid. Data is invalid for plotting.")

    regression_result = fit_regression_line(x_valid, y_valid, poly_degree)

    # Regression line
    x_line = np.linspace(lower, upper, 100)
    y_line = np.polyval(regression_result.coefficients[::-1], x_line) + regression_result.intercept
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line (Degree {poly_degree})')

    # Overall statistics
    overall_stats = get_statistics(x_valid, y_valid)

    # Display stats
    plt.text(0.05, 0.95, f"Pearson Correlation: {overall_stats.pearson:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5))
    plt.text(0.05, 0.90, f"MSE: {regression_result.mse:.4f}", transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5))
    if poly_degree == 1:
        plt.text(0.05, 0.85, f"Regression Coefficients: {regression_result.coefficients[1]:.2f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round", alpha=0.5))

    # Per-bin regression and statistics
    per_bin_stats = []
    if plot_bin_regression or return_results:
        for i, bin_y in enumerate(binned_y):
            bin_mask = (x_valid > bin_centers[i] - (upper - lower) / (2 * num_bins)) & (
                    x_valid <= bin_centers[i] + (upper - lower) / (2 * num_bins))
            bin_x = x_valid[bin_mask]
            bin_y = y_valid[bin_mask]

            if len(bin_x) > 1:  # Perform regression only if bin has enough data
                bin_regression_result = fit_regression_line(bin_x, bin_y, poly_degree=1)
                bin_stats = get_statistics(bin_x, bin_y)
                per_bin_stats.append({
                    "bin_index": i + 1,
                    "bin_center": bin_centers[i],
                    "bin_stats": bin_stats,
                    "regression": bin_regression_result
                })

                if plot_bin_regression:
                    plt.plot(bin_x, bin_regression_result.predictions.reshape(-1, 1),
                             label=f"Bin {i + 1} coeff: {bin_regression_result.coefficients[1]:.2f}")

                if verbose:
                    print(f"""
                    Statistics of bin {i + 1}:
                      Pearson Correlation: {bin_stats.pearson:.2f}
                      Spearman Correlation: {bin_stats.spearman:.2f}
                      Standard Deviation: {bin_stats.std:.2f}
                      Mean: {bin_stats.mean:.2f}
                      Median: {bin_stats.median:.2f}
                      Number of Data Points: {bin_stats.n_points}
                      Regression Coefficients: {bin_regression_result.coefficients[1]:.2f}
                      MSE: {bin_regression_result.mse:.4f}
                    """)

    if verbose:
        print(f"""
        Overall Statistics:
          Pearson Correlation: {overall_stats.pearson:.2f}
          Spearman Correlation: {overall_stats.spearman:.2f}
          Standard Deviation: {overall_stats.std:.2f}
          Mean: {overall_stats.mean:.2f}
          Median: {overall_stats.median:.2f}
          Regression Coefficients: {[round(coeff, 2) for coeff in regression_result.coefficients]}
        """)

    plt.xticks(bin_centers, [round(center, 2) for center in bin_centers])
    plt.title(title)
    plt.xlabel(x_label)
    plt.xlim(lower, upper)
    plt.ylabel(y_label)
    plt.ylim(*y_lim)
    plt.legend()
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()

    if return_results:
        return {
            "overall_statistics": overall_stats,
            "regression_result": regression_result,
            "per_bin_statistics": per_bin_stats
        }


def plot_scatter_with_regression(x: np.ndarray,
                                 y: np.ndarray,
                                 x_lim: tuple = (0, 1),
                                 y_lim: tuple = (0, 1),
                                 title: str = "Scatterplot with Regression",
                                 x_label: str = "IoU Difference",
                                 y_label: str = "Similarity Score",
                                 save_fig_path: str = None) -> None:
    """
    Plot a scatterplot of x and y, and add a regression line over the scatter.

    :param x: Independent variable values (numpy array).
    :param y: Dependent variable values (numpy array).
    :param x_lim: Limits for the x-axis.
    :param y_lim: Limits for the y-axis.
    :param title: Title of the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param save_fig_path: Path to save the figure. If None, the figure is displayed.
    :returns: None
    """
    lower, upper = x_lim
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid]
    y_valid = y[valid]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_valid, y_valid, alpha=0.6, label="Data points")

    if len(x_valid) > 1:  # Ensure enough data points are available
        reg = LinearRegression().fit(x_valid.reshape(-1, 1), y_valid)
        coeff = reg.coef_[0]
        x_line = np.linspace(lower, upper, 100).reshape(-1, 1)
        y_line = reg.predict(x_line)
        plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line, Coefficient: {coeff:.2f}')
    else:
        print("Insufficient data points for regression.")

    plt.title(title)
    plt.xlabel(x_label)
    plt.xlim(*x_lim)
    plt.ylabel(y_label)
    plt.ylim(*y_lim)
    plt.legend()
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()

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

def list_is_unique(lst: list) -> bool:
    """
    Check if all elements in a list are unique.

    :param lst: List to check

    :return: True if all elements are unique, False otherwise
    """
    if len(lst) <= 1:
        return True
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j]:
                return False
    return True

def convert_to_integers(list_of_tuples: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """
    Convert all elements in a list of tuples to integers.

    :param list_of_tuples: List of tuples with float values

    :return: List of tuples with integer values
    """
    return [(int(x), int(y)) for x, y in list_of_tuples]


def standardize_data(data: np.ndarray, axis: int=0) -> np.ndarray:
    """
    Standardize the given data using the formula: (x - mean) / std.

    :param data: Input data
    :param axis: Axis along which to standardize the data (for row-wise standardization, use axis=0. For column-wise standardization, use axis=1)

    :return: Standardized data
    """
    return (data - np.mean(data, axis=axis, keepdims=True)) / np.std(data, axis=axis, keepdims=True)


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


def copy_or_move_images(image_paths: list[str], directory: str, operation: str="copy") -> None:
    """
    Move or copy images to the specified directory.

    :param image_paths: List of image paths
    :param directory: Directory to move or copy the images
    :param operation: Operation to perform. Choose from ['copy', 'cut']. Default is 'copy'.
    """
    assert operation in ['copy', 'cut'], "Invalid operation. Choose from ['copy', 'cut']"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for image in image_paths:
        if operation == "copy":
            shutil.copy(image, directory)
        elif operation == "cut":
            shutil.move(image, directory)

def average(matrix: np.ndarray | torch.Tensor) -> float:
    """
    Compute the average of the given matrix.

    :param matrix: Input matrix

    :return: Average value
    """
    return np.mean(matrix) if isinstance(matrix, np.ndarray) else torch.mean(matrix).item()


@check_is_image()
def gaussian_blur(image: np.ndarray | torch.Tensor, kernel_size: int=None, sigma: float=1.0) -> np.ndarray | torch.Tensor:
    """
    Apply Gaussian blurring to the given image.

    :param image: Input image
    :param kernel_size: Size of the kernel
    :param sigma: Standard deviation of the kernel

    :return: Blurred image
    """
    if not kernel_size:
        kernel_size = 2 * int(3 * sigma) + 1
    min_kernel_size = 2 * int(3 * sigma) + 1
    max_kernel_size = 2 * int(5 * sigma) + 1
    if not min_kernel_size <= kernel_size <= max_kernel_size:
        raise ValueError(f"Kernel radius must be between 2 * 3-5 times the standard deviation plus one. " 
                         f"In this case, it should be between {min_kernel_size} and {max_kernel_size} "
                         f"Got kernel size: {kernel_size}")
    if isinstance(image, np.ndarray):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif torch.is_tensor(image):
        return TF.gaussian_blur(image, kernel_size, sigma).clip(0.0, 1.0)


@check_is_image()
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


