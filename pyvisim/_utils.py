from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    rand_score,
)
from sklearn.metrics.pairwise import cosine_similarity as cs


def read_image_rgb(path: str) -> np.ndarray:
    """
    Read an image from disk and convert it to RGB.

    :param path: Path to the image file.
    :return: Image as a NumPy array (H, W, C) in RGB order.
    :raises FileNotFoundError: If the image cannot be read from the given path.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{path}'.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
        raise ValueError(
            f"Cosine similarity requires at least 2 features. Got {x.shape[-1]} features for x and {y.shape[-1]} features for y."
        )

    return np.asarray(cs(x, y))


def plot_image(image: np.ndarray | torch.Tensor, title: str = "Image") -> None:
    """
    Plot a single image.

    :param image: Image as a NumPy array (H, W, C) or torch tensor (C, H, W)
    :param title: Title of the plot
    """
    plt.figure(figsize=(10, 10))
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def cluster_and_return_labels(
    data: np.ndarray,
    method: Literal["kmeans", "dbscan", "spectral"] = "kmeans",
    n_clusters: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Clusters 'data' using the specified method.

    :param data: A 2D NumPy array of shape (N, D)
    :param method: 'kmeans', 'dbscan', or 'spectral'
    :param n_clusters: Number of clusters (if applicable)
    :param kwargs: Additional arguments to pass to the clustering constructor
    :return: 1D NumPy array of cluster labels (shape: (N,))
    """
    if method == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans.")
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        return np.asarray(model.fit_predict(data))

    if method == "dbscan":
        # DBSCAN doesn't need n_clusters (but can accept eps, min_samples)
        model = DBSCAN(**kwargs)
        return np.asarray(model.fit_predict(data))

    if method == "spectral":
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Spectral Clustering.")
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            random_state=42,
            **kwargs,
        )
        return np.asarray(model.fit_predict(data))

    raise ValueError(f"Unknown method: {method}")


def cluster_images_and_generate_statistics(
    features: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    method: Literal["kmeans", "dbscan", "spectral"] = "kmeans",
    **kwargs: Any,
) -> dict[str, float]:
    """
    Clusters the given features and computes RI, ARI and NMI.

    :param features: (N, D) array of feature vectors
    :param true_labels: (N,) array of ground truth class labels
    :param n_clusters: Number of clusters to find
    :param method: 'kmeans', 'dbscan', or 'spectral'
    :param kwargs: Additional parameters for the clustering method
    :return: Dictionary of statistics {'ri': ..., 'ari': ..., 'nmi': ...}
    """
    cluster_labels = cluster_and_return_labels(
        data=features,
        method=method,
        n_clusters=n_clusters if method != "dbscan" else None,
        **kwargs,
    )

    return {
        "ri": rand_score(true_labels, cluster_labels),
        "ari": adjusted_rand_score(true_labels, cluster_labels),
        "nmi": adjusted_mutual_info_score(true_labels, cluster_labels),
    }


def plot_and_save_heatmap(
    matrix: list[Any] | np.ndarray | torch.Tensor,
    figsize: tuple[int, int] | None = None,
    x_tick_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    cbar_kws: dict[str, str] | None = None,
    title: str = "Heatmap",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    show: bool = True,
    save_fig_path: str | None = None,
) -> None:
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
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    elif isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    figsize = (
        (
            matrix.shape[1] * 0.7,
            matrix.shape[0] * 0.7,
        )
        if figsize is None
        else figsize
    )
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=x_tick_labels if x_tick_labels else list(range(matrix.shape[1])),
        yticklabels=y_tick_labels if y_tick_labels else list(range(matrix.shape[0])),
        cbar_kws=cbar_kws if cbar_kws else {"label": "value"},
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_fig_path:
        plt.savefig(save_fig_path)
    if show:
        plt.show()
    plt.close()


def plot_and_save_barplot(
    data: dict[str, list[float]],
    bar_labels: list[str],
    title: str = "Barplot",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    save_path: str | None = None,
    show: bool = True,
) -> None:
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
        raise ValueError(
            "All lists in data must have the same length as the number of bar labels."
        )

    x = np.arange(len(x_labels))  # the label locations
    width = 0.8 / num_groups  # width of each bar

    plt.figure(figsize=(10, 6))

    for i in range(num_groups):
        heights = [v[i] for v in values]
        plt.bar(x + i * width, heights, width, label=bar_labels[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x + width * (num_groups - 1) / 2, x_labels)  # Center the tick labels
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()
