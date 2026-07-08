"""Download, cache and verify OpenAI's official CLIP TorchScript checkpoints.

The checkpoint registry mirrors OpenAI's reference implementation
(https://github.com/openai/CLIP) and the ``openai`` pretrained tag of
`open_clip <https://github.com/mlfoundations/open_clip>`_: each checkpoint is
served from OpenAI's public bucket under a URL path segment that is the
SHA-256 digest of the file itself. Both cache hits and fresh downloads are
verified against that digest before a path is handed to
:func:`torch.jit.load`.

Checkpoints are cached in ``~/.cache/clip`` by default -- the directory also
used by ``open_clip`` and OpenAI's ``clip`` package -- so weights already
downloaded by either library are reused instead of re-downloaded.

This module is a low-level library file: it raises exceptions
(:class:`ValueError`, :class:`OSError`,
:class:`~pyvisim._errors.ChecksumMismatchError`,
:class:`requests.RequestException`) and leaves handling to callers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from ..._errors import ChecksumMismatchError

_logger = logging.getLogger("CLIP_Embedder")

#: Root of the public bucket OpenAI serves its original CLIP checkpoints from.
_OPENAI_CHECKPOINT_ROOT = "https://openaipublic.azureedge.net/clip/models"

#: Default cache directory, shared with ``open_clip`` and OpenAI's ``clip``.
_DEFAULT_CACHE_DIR = Path("~/.cache/clip")

_DOWNLOAD_CHUNK_SIZE = 8192
_DOWNLOAD_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CheckpointSpec:
    """
    Static description of one official OpenAI CLIP checkpoint.

    :param filename: File name of the TorchScript archive in OpenAI's bucket.
    :param sha256: Expected SHA-256 hex digest of the archive. OpenAI embeds
        this digest in the download URL, so it also selects the file served.
    :param image_size: Side length in pixels of the square model input.
    :param embedding_dim: Dimensionality of the image embedding space.
    """

    filename: str
    sha256: str
    image_size: int
    embedding_dim: int

    @property
    def url(self) -> str:
        """Direct download URL of the checkpoint archive."""
        return f"{_OPENAI_CHECKPOINT_ROOT}/{self.sha256}/{self.filename}"


#: Supported CLIP variants. URLs and digests match OpenAI's reference
#: implementation and open_clip's ``openai`` pretrained tag; embedding
#: dimensions and input sizes match the corresponding open_clip model configs.
CHECKPOINTS: dict[str, CheckpointSpec] = {
    "ViT-B/32": CheckpointSpec(
        filename="ViT-B-32.pt",
        sha256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        image_size=224,
        embedding_dim=512,
    ),
    "ViT-B/16": CheckpointSpec(
        filename="ViT-B-16.pt",
        sha256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        image_size=224,
        embedding_dim=512,
    ),
    "ViT-L/14": CheckpointSpec(
        filename="ViT-L-14.pt",
        sha256="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        image_size=224,
        embedding_dim=768,
    ),
}


def available_variants() -> tuple[str, ...]:
    """
    List the names of the supported CLIP variants.

    :return: The variant names accepted by :func:`get_checkpoint_spec`.
    """
    return tuple(CHECKPOINTS)


def get_checkpoint_spec(variant: str) -> CheckpointSpec:
    """
    Look up the checkpoint description of a CLIP variant.

    :param variant: Official OpenAI variant name, e.g. ``"ViT-B/32"``.
    :return: The matching :class:`CheckpointSpec`.
    :raises ValueError: If ``variant`` is not a supported variant.
    """
    try:
        return CHECKPOINTS[variant]
    except KeyError:
        raise ValueError(
            f"Unsupported CLIP variant: {variant!r}. "
            f"Supported variants: {', '.join(CHECKPOINTS)}."
        ) from None


def fetch_checkpoint(variant: str, cache_dir: str | Path | None = None) -> Path:
    """
    Return a verified local path to the checkpoint of a CLIP variant.

    The cache is consulted first: a cached file whose SHA-256 digest matches
    the published one is returned as-is, and a corrupted cached file triggers
    a warning and a re-download. A fresh download is verified before it
    replaces anything in the cache, so this function never returns a path to
    an unverified file.

    :param variant: Official OpenAI variant name, e.g. ``"ViT-B/32"``.
    :param cache_dir: Directory the checkpoint is cached in. Defaults to
        ``~/.cache/clip``, the directory also used by ``open_clip`` and
        OpenAI's ``clip`` package, so existing downloads are reused.
    :return: Path to the verified TorchScript archive.
    :raises ValueError: If ``variant`` is not a supported variant.
    :raises OSError: If the cache path is obstructed by a non-regular file.
    :raises ChecksumMismatchError: If the downloaded file fails verification.
    :raises requests.RequestException: If the download itself fails.
    """
    spec = get_checkpoint_spec(variant)
    cache = Path(cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR)
    cache = cache.expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / spec.filename

    if target.exists() and not target.is_file():
        raise OSError(f"{target} exists but is not a regular file.")
    if target.is_file():
        if _sha256_digest(target) == spec.sha256:
            return target
        warnings.warn(
            f"Cached checkpoint {target} failed its SHA-256 integrity check; "
            f"re-downloading it.",
            FutureWarning,
            stacklevel=2,
        )
    _download_checkpoint(spec, target)
    return target


def _sha256_digest(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file, reading it in chunks.

    :param path: Path of the file to hash.
    :return: The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checkpoint(spec: CheckpointSpec, target: Path) -> None:
    """
    Download a checkpoint to ``target``, verifying its SHA-256 digest.

    The archive is streamed into a unique temporary file next to ``target``
    and hashed on the fly; only after the digest matches is it atomically
    moved into place. A failed or interrupted download therefore never
    leaves a partial or unverified file in the cache.

    :param spec: Description of the checkpoint to download.
    :param target: Final path of the verified archive.
    :raises ChecksumMismatchError: If the downloaded bytes do not match the
        published digest.
    :raises requests.RequestException: If the HTTP request fails.
    """
    _logger.info("Downloading %s to %s", spec.url, target)
    response = requests.get(spec.url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    digest = hashlib.sha256()
    descriptor, partial_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".partial"
    )
    partial = Path(partial_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as handle,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {target.name}",
            ) as progress_bar,
        ):
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    progress_bar.update(len(chunk))
        if digest.hexdigest() != spec.sha256:
            raise ChecksumMismatchError(
                f"Checkpoint downloaded from {spec.url} failed its SHA-256 "
                f"integrity check: expected {spec.sha256}, "
                f"got {digest.hexdigest()}."
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    _logger.info("Downloaded and verified %s", target)
