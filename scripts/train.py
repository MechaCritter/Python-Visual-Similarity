"""
This module provides functionalities to train models using K-Means clustering,
Gaussian Mixture Models (GMM), and Principal Component Analysis (PCA) on given input features.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

RANDOM_STATE = 42


def train_k_means(features: np.ndarray, n_clusters: int, **kwargs) -> KMeans:
    """
    Train a K-Means clustering model.

    :param features: feature matrix of shape (n_samples, n_features)
    :param n_clusters: number of clusters
    :param kwargs: Additional arguments for the K-Means algorithm.
    :return: A trained KMeans model.
    """
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, **kwargs).fit(features)


def train_pca(features: np.ndarray, reduction_factor: int = 2, standardize: bool = False, **kwargs) -> PCA:
    """
    Train a Principal Component Analysis (PCA) model after standardizing features.
    :param features: feature matrix of shape (n_samples, n_features)
    :param reduction_factor:
    :param standardize: Whether to standardize the features to have zero mean and unit variance.
    :return: A trained PCA model.
    """
    if standardize:
        from sklearn.preprocessing import StandardScaler
        features = StandardScaler().fit_transform(features)
    return PCA(n_components=features.shape[1] // reduction_factor, random_state=RANDOM_STATE, **kwargs).fit(features)


def train_gmm(features: np.ndarray, n_components: int, **kwargs) -> GaussianMixture:
    """
    Train a Gaussian Mixture Model (GMM).

    :param features: feature matrix of shape (n_samples, n_features)
    :param n_components: number of mixture components
    :param kwargs: Additional arguments for the GaussianMixture algorithm.
    :return: A trained GaussianMixture model.
    """
    return GaussianMixture(n_components=n_components, random_state=RANDOM_STATE, covariance_type='diag', **kwargs).fit(features)
