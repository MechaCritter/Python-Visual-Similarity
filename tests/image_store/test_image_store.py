"""Tests for :class:`pyvisim.image_store.InMemoryImageEmbeddingStore`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.encoders import VLADEncoder
from pyvisim.image_store import InMemoryImageEmbeddingStore


@pytest.fixture(scope="module")
def gallery_paths(
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> list[str]:
    """Write the training images to disk and return their paths.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :param category_train_images_flat: corner-rich training images.
    :returns: one ``.png`` path per training image.
    """
    directory = tmp_path_factory.mktemp("store_gallery")
    paths: list[str] = []
    for index, image in enumerate(category_train_images_flat):
        rgb = np.stack([image, image, image], axis=-1)
        path = directory / f"img_{index}.png"
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def store(
    gallery_paths: list[str],
    learned_vlad_encoder: VLADEncoder,
) -> InMemoryImageEmbeddingStore:
    """An :class:`InMemoryImageEmbeddingStore` over the on-disk gallery.

    :param gallery_paths: the gallery image paths.
    :param learned_vlad_encoder: a fitted VLAD encoder (PCA and non-PCA variants).
    :returns: a store backed by an exact (L2, IVF-Flat) index.
    """
    return InMemoryImageEmbeddingStore(
        gallery_paths,
        learned_vlad_encoder,
        "ivf-flat",
        index_params={"nlist": 4, "nprobe": 4},
    )


def test_save_appends_safetensors_suffix(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The save path gains a ``.safetensors`` suffix when missing."""
    target = tmp_path_factory.mktemp("save_suffix") / "mystore"
    written = store.save_to_disk(target)
    assert written.suffix == ".safetensors"
    assert written.exists()


def test_save_load_preserves_paths(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store keeps the gallery paths in order."""
    target = tmp_path_factory.mktemp("rt_paths") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))
    assert loaded.paths == store.paths


def test_save_load_preserves_index_config(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store keeps the index type, metric and params."""
    target = tmp_path_factory.mktemp("rt_cfg") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))
    assert loaded.index_type == store.index_type
    assert loaded.quantizer == store.quantizer
    assert loaded.index_params == store.index_params


def test_save_load_preserves_embeddings(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The reloaded store's embeddings match the original exactly (L2 index)."""
    target = tmp_path_factory.mktemp("rt_emb") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))
    assert loaded.embeddings.shape == store.embeddings.shape
    assert np.allclose(loaded.embeddings, store.embeddings, atol=1e-6)


def test_save_does_not_mutate_store(
    store: InMemoryImageEmbeddingStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Serialising the store leaves its own state untouched."""
    before = store.embeddings.copy()
    target = tmp_path_factory.mktemp("no_mutate") / "store.safetensors"
    store.save_to_disk(target)
    assert np.array_equal(store.embeddings, before)
    assert store.paths == store.paths


def test_save_load_preserves_encoder(
    store: InMemoryImageEmbeddingStore,
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """The reloaded store carries an equivalent encoder.

    The reconstructed encoder is compared against the store's own encoder
    behaviourally (same image encodes to the same vector) and against the same
    encoder serialised on its own with ``save_to_disk``.
    """
    target = tmp_path_factory.mktemp("rt_encoder")
    loaded = InMemoryImageEmbeddingStore.load_from_disk(
        store.save_to_disk(target / "store.safetensors")
    )
    assert isinstance(loaded.encoder, VLADEncoder)

    # The store's encoder, serialised on its own, reloaded from disk.
    encoder_path = store.encoder.save_to_disk(target / "encoder")
    directly_loaded = VLADEncoder.load_from_disk(encoder_path)

    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    from_store = store.encoder.encode(probe)
    from_loaded_store = loaded.encoder.encode(probe)
    from_direct = directly_loaded.encode(probe)

    assert np.allclose(from_loaded_store, from_store, atol=1e-5)
    assert np.allclose(from_loaded_store, from_direct, atol=1e-5)


def test_load_missing_file_raises(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Loading a non-existent store file raises ``FileNotFoundError``."""
    missing = tmp_path_factory.mktemp("missing") / "absent.safetensors"
    with pytest.raises(FileNotFoundError):
        InMemoryImageEmbeddingStore.load_from_disk(missing)
