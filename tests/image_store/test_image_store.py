"""Tests for :class:`pyvisim.image_store.InMemoryImageEmbeddingStore`."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pyvisim.encoders import Pipeline, VLADEncoder
from pyvisim.functional import Candidate
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
def large_gallery_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> list[str]:
    """160 corner-rich images for IVF-PQ tests that need many training vectors.

    :param tmp_path_factory: pytest's session temp-directory factory.
    :returns: 160 ``.png`` paths, enough for ``nbits=2`` IVF-PQ training
        (requires ``39 * 2**2 = 156`` vectors).
    """
    rng = np.random.default_rng(0)
    directory = tmp_path_factory.mktemp("ivf_pq_gallery")
    base = np.zeros((256, 256), dtype=np.uint8)
    for row in range(0, 256, 16):
        for col in range(0, 256, 16):
            if (row // 16 + col // 16) % 2 == 0:
                base[row : row + 16, col : col + 16] = 255
    paths: list[str] = []
    for i in range(160):
        noise = rng.integers(-10, 11, size=base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        rgb = np.stack([img, img, img], axis=-1)
        path = directory / f"img_{i}.png"
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


# Construction and exposed state


def test_reports_paths_dim_and_len(
    store: InMemoryImageEmbeddingStore, gallery_paths: list[str]
) -> None:
    """The store exposes the gallery paths, dimensionality and size."""
    assert store.paths == gallery_paths
    assert len(store) == len(gallery_paths)
    assert store.dim == store.embeddings.shape[1]
    assert store.index_type == "ivf-flat"


def test_contains_and_repr(store: InMemoryImageEmbeddingStore) -> None:
    """``in`` checks gallery membership and ``repr`` names the store."""
    assert store.paths[0] in store
    assert "absent.png" not in store
    assert "InMemoryImageEmbeddingStore(" in repr(store)


def test_search_returns_expected_shapes(
    store: InMemoryImageEmbeddingStore,
) -> None:
    """``search`` returns ``(M, k)`` score and id arrays."""
    scores, ids = store.search(store.embeddings[:3], k=4)
    assert scores.shape == (3, 4)
    assert ids.shape == (3, 4)


def test_retrieve_recovers_self(
    store: InMemoryImageEmbeddingStore,
    gallery_paths: list[str],
    category_train_images_flat: list[np.ndarray],
) -> None:
    """Retrieving with a gallery image returns that image as the top match."""
    gray = category_train_images_flat[2]
    probe = np.stack([gray, gray, gray], axis=-1)
    results = store.retrieve_top_k_similar(probe, k=3)
    assert isinstance(results[0][0], Candidate)
    assert results[0][0].path == gallery_paths[2]


def test_unknown_index_type_raises(
    gallery_paths: list[str], learned_vlad_encoder: VLADEncoder
) -> None:
    """An unknown index_type is rejected before any image is encoded."""
    with pytest.raises(ValueError, match="Unknown index_type"):
        InMemoryImageEmbeddingStore(gallery_paths, learned_vlad_encoder, "bogus")


@pytest.mark.parametrize("index_type", ["hnsw", "int8"])
def test_unimplemented_index_types_raise(
    index_type: str, gallery_paths: list[str], learned_vlad_encoder: VLADEncoder
) -> None:
    """The sketched index types raise ``NotImplementedError`` when built."""
    with pytest.raises(NotImplementedError):
        InMemoryImageEmbeddingStore(gallery_paths[:6], learned_vlad_encoder, index_type)


def test_non_string_path_raises(learned_vlad_encoder: VLADEncoder) -> None:
    """A non-string path is rejected before any encoding happens."""
    with pytest.raises(TypeError, match="Image paths must be strings"):
        InMemoryImageEmbeddingStore([123], learned_vlad_encoder)  # type: ignore[list-item]


def test_missing_file_raises(
    learned_vlad_encoder: VLADEncoder, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A missing image file aborts construction by default."""
    missing = str(tmp_path_factory.mktemp("missing") / "gone.png")
    with pytest.raises(FileNotFoundError):
        InMemoryImageEmbeddingStore([missing], learned_vlad_encoder)


def test_skip_errors_warns_and_keeps_good_images(
    gallery_paths: list[str],
    learned_vlad_encoder: VLADEncoder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``skip_errors`` warns about and omits images that fail to encode."""
    missing = str(tmp_path_factory.mktemp("partial") / "gone.png")
    with pytest.warns(FutureWarning, match="Skipped 1 image"):
        store = InMemoryImageEmbeddingStore(
            [*gallery_paths[:5], missing],
            learned_vlad_encoder,
            index_params={"nlist": 2, "nprobe": 2},
            skip_errors=True,
        )
    assert store.paths == gallery_paths[:5]


def test_all_images_unreadable_raises(
    learned_vlad_encoder: VLADEncoder, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A store cannot be built when every image fails to encode."""
    missing = str(tmp_path_factory.mktemp("empty") / "gone.png")
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="No images could be encoded"):
            InMemoryImageEmbeddingStore(
                [missing], learned_vlad_encoder, skip_errors=True
            )


def test_inner_product_embeddings_are_normalized(
    gallery_paths: list[str], learned_vlad_encoder: VLADEncoder
) -> None:
    """An inner-product store exposes L2-normalised embeddings."""
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:8],
        learned_vlad_encoder,
        "ivf-flat",
        quantizer="inner_product",
        index_params={"nlist": 2, "nprobe": 2},
    )
    norms = np.linalg.norm(store.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_ivf_pq_store_builds_and_searches(
    large_gallery_paths: list[str], learned_vlad_encoder: VLADEncoder
) -> None:
    """An IVF-PQ store builds and searches correctly."""
    store = InMemoryImageEmbeddingStore(
        large_gallery_paths,
        learned_vlad_encoder,
        "ivf-pq",
        index_params={"nlist": 4, "nprobe": 4, "m": 8, "nbits": 2},
    )
    scores, ids = store.search(store.embeddings[:2], k=3)
    assert scores.shape == (2, 3)
    assert ids.shape == (2, 3)


def test_store_with_pipeline_encoder_round_trips(
    gallery_paths: list[str],
    learned_vlad_encoder: VLADEncoder,
    tmp_path_factory: pytest.TempPathFactory,
    category_train_images_flat: list[np.ndarray],
) -> None:
    """A store built on a Pipeline serialises and reconstructs the Pipeline."""
    pipeline = Pipeline([learned_vlad_encoder])
    store = InMemoryImageEmbeddingStore(
        gallery_paths[:8],
        pipeline,
        index_params={"nlist": 2, "nprobe": 2},
    )
    target = tmp_path_factory.mktemp("pipeline_store") / "store.safetensors"
    loaded = InMemoryImageEmbeddingStore.load_from_disk(store.save_to_disk(target))

    assert isinstance(loaded.encoder, Pipeline)
    gray = category_train_images_flat[0]
    probe = np.stack([gray, gray, gray], axis=-1)
    assert np.allclose(
        loaded.encoder.encode(probe), store.encoder.encode(probe), atol=1e-5
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
