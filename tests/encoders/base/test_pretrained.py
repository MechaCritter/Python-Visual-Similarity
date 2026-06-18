"""Tests for loading bundled pretrained encoders via ``from_pretrained``."""

from __future__ import annotations

import pytest

from pyvisim.encoders import (
    FisherVectorEncoder,
    PretrainedFisher,
    PretrainedVLAD,
    VLADEncoder,
)
from pyvisim.features import SIFT, RootSIFT


def test_from_pretrained_vlad_rootsift() -> None:
    """A pretrained RootSIFT VLAD encoder loads with a fitted k=256 K-Means."""
    encoder = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT)
    assert isinstance(encoder, VLADEncoder)
    assert isinstance(encoder.feature_extractor, RootSIFT)
    assert encoder.clustering_model.is_fitted
    assert encoder.clustering_model.n_clusters == 256
    assert encoder.pca is None


def test_from_pretrained_vlad_pca_restores_pca() -> None:
    """The PCA variant restores a fitted PCA reducing RootSIFT 128 -> 64 dims."""
    encoder = VLADEncoder.from_pretrained(PretrainedVLAD.OXFORD102_K256_ROOTSIFT_PCA)
    assert encoder.pca is not None
    assert encoder.pca.is_fitted
    assert encoder.pca.n_components == 64


def test_from_pretrained_fisher_sift() -> None:
    """A pretrained SIFT Fisher encoder loads with a fitted k=256 GMM."""
    encoder = FisherVectorEncoder.from_pretrained(PretrainedFisher.OXFORD102_K256_SIFT)
    assert isinstance(encoder, FisherVectorEncoder)
    assert isinstance(encoder.feature_extractor, SIFT)
    assert encoder.clustering_model.is_fitted
    assert encoder.clustering_model.n_clusters == 256


def test_from_pretrained_wrong_encoder_type_raises() -> None:
    """Loading a Fisher pretrained file as a VLAD encoder raises ``ValueError``."""
    with pytest.raises(ValueError, match="was saved by FisherVectorEncoder"):
        VLADEncoder.from_pretrained(PretrainedFisher.OXFORD102_K256_ROOTSIFT)
