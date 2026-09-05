"""
Measure what the reading threads of ``InMemoryImageEmbeddingStore`` buy.

The same gallery of Oxford Flower images is read and decoded once per worker
count, and the wall-clock time of each run is written to a markdown report. Only
the reading stage is timed: no embedder runs, so the numbers describe the stage
``num_workers`` actually controls rather than the work it is meant to hide
behind.

Run it with::

    uv run python scripts/benchmark_num_workers.py
"""

import argparse
import gc
import os
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from pyvisim.datasets import OxfordFlowerDataset

# The reading stage is internal to the store, which exposes no way to run it
# without embedding. Driving it directly is what keeps the embedder out of the
# measurement.
from pyvisim.image_store.image_store import _decoded_images

_SCRIPT_REF = "scripts/benchmark_num_workers.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Worker counts the gallery is read at, in the order they are run.
_WORKER_COUNTS = (1, 2, 4, 8, 16)

_DATASET_SPLIT = "train"
#: Times every worker count is measured. The counts are interleaved across the
#: rounds, so clock drift and thermal throttling reach all of them alike
#: instead of biasing whichever one would have run last.
_ROUNDS = 5
#: Batch size the prefetch window is derived from, standing in for the batch
#: size of an embedder the store would be built with.
_BATCH_SIZE = 32
#: Batches the reading threads may run ahead, as in the store's own default.
_NUM_PREFETCH_BATCHES = 4


def _log(message: str) -> None:
    """Print a progress line right away, so a long run can be followed."""
    print(message, flush=True)


@dataclass(frozen=True)
class Measurement:
    """
    What one worker count cost.

    :param num_workers: Threads the gallery was read on.
    :param rounds: Wall-clock duration of every round, in seconds.
    """

    num_workers: int
    rounds: list[float]

    @property
    def seconds(self) -> float:
        """Median duration of the rounds, in seconds."""
        return statistics.median(self.rounds)

    @property
    def fastest(self) -> float:
        """Duration of the fastest round, in seconds."""
        return min(self.rounds)


def _gallery_paths(count: int, seed: int) -> list[str]:
    """
    Draw the gallery image paths from the Oxford Flower dataset.

    :param count: Number of images the gallery holds.
    :param seed: Seed of the image sampling.
    :return: The sampled image file paths.
    :raises ValueError: If the split holds fewer images than requested.
    """
    dataset = OxfordFlowerDataset(purpose=_DATASET_SPLIT)
    if count > len(dataset):
        raise ValueError(
            f"The {_DATASET_SPLIT!r} split holds {len(dataset)} images, but "
            f"{count} were requested."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=count, replace=False)
    return [dataset.image_paths[int(index)] for index in indices]


def _warm_page_cache(paths: Sequence[str]) -> None:
    """
    Read every gallery file once, before any measurement.

    A first read pulls the file off the disk, every later one finds it in the
    page cache of the operating system. Warming them all up front keeps that
    one-off cost out of the measurements, which compare how well the decoding
    itself parallelises.
    """
    for path in paths:
        with open(path, "rb") as handle:
            handle.read()


def _read_gallery(paths: list[str], num_workers: int, prefetch: int) -> float:
    """
    Read and decode the whole gallery once, and return how long it took.

    :param paths: Gallery image paths to read.
    :param num_workers: Threads decoding the image files.
    :param prefetch: Images the decoding threads may read ahead.
    :return: The wall-clock duration of the read, in seconds.
    :raises ValueError: If an image could not be decoded.
    """
    failures: list[str] = []
    gc.collect()
    start = time.perf_counter()
    for _ in _decoded_images(paths, num_workers, prefetch, False, failures):
        pass
    return time.perf_counter() - start


def _measure(
    paths: list[str],
    worker_counts: Sequence[int],
    rounds: int,
    prefetch: int,
) -> list[Measurement]:
    """
    Read the gallery at every worker count, once per round.

    :param paths: Gallery image paths to read.
    :param worker_counts: Thread counts to measure.
    :param rounds: Times every worker count is measured.
    :param prefetch: Images the decoding threads may read ahead.
    :return: One measurement per worker count, in the given order.
    """
    timings: dict[int, list[float]] = {count: [] for count in worker_counts}
    for round_index in range(rounds):
        for num_workers in worker_counts:
            seconds = _read_gallery(paths, num_workers, prefetch)
            timings[num_workers].append(seconds)
            _log(
                f"round {round_index + 1}/{rounds}, {num_workers:>2} worker(s): "
                f"{seconds:.3f}s"
            )
    return [Measurement(count, timings[count]) for count in worker_counts]


def _format_ratio(value: float, baseline: float) -> str:
    """Format a ratio against a baseline, or ``n/a`` when there is nothing to divide by."""
    if baseline <= 0.0:
        return "n/a"
    return f"{value / baseline:.2f}x"


def _format_table(measurements: Sequence[Measurement], count: int) -> str:
    """Render the measurements as a markdown table."""
    lines = [
        "| Workers | Time (s) | Fastest (s) | Images/s | Speed-up | Efficiency |",
        "|---|---|---|---|---|---|",
    ]
    single = measurements[0]
    for measurement in measurements:
        speed_up = single.seconds / measurement.seconds
        lines.append(
            f"| {measurement.num_workers} | {measurement.seconds:.3f} "
            f"| {measurement.fastest:.3f} | {count / measurement.seconds:.0f} "
            f"| {_format_ratio(single.seconds, measurement.seconds)} "
            f"| {speed_up / measurement.num_workers:.0%} |"
        )
    return "\n".join(lines)


def _best(measurements: Sequence[Measurement]) -> Measurement:
    """Return the fastest measurement."""
    return min(measurements, key=lambda measurement: measurement.seconds)


def _package_version(name: str) -> str:
    """Resolve an installed package version, or ``"unknown"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_report(
    measurements: Sequence[Measurement],
    count: int,
    rounds: int,
    prefetch: int,
    seed: int,
) -> str:
    """Render the complete markdown report."""
    single, best = measurements[0], _best(measurements)
    worker_counts = ", ".join(
        str(measurement.num_workers) for measurement in measurements
    )
    return f"""# Reading threads benchmark

> [!IMPORTANT]
> This file was generated by the script [`{_SCRIPT_REF}`](../../{_SCRIPT_REF}).
> **Do not edit manually!**

The same {count} images of the Oxford Flower dataset (`{_DATASET_SPLIT}` split,
seed {seed}) are read and decoded once per worker count ({worker_counts}), the
way `InMemoryImageEmbeddingStore` reads a gallery it is built on. Nothing is
embedded: the measurement isolates the stage `num_workers` controls. In a real
build the embedder runs alongside it, so the end-to-end gain is at most the one
below and is often smaller, because the reads are meant to hide behind the
embedder's own work.

Every worker count is measured {rounds} times and the table reports the median,
with the counts interleaved across the rounds so that clock drift reaches all of
them alike. Every file is read once before the first measurement, which leaves
the images in the page cache of the operating system: the table compares how
well the decoding parallelises, not how fast the disk is. The **speed-up** is
the time of the single-threaded read divided by the one of the run, and the
**efficiency** is that speed-up divided by the worker count, so 100% is a
perfectly linear gain.

Decoding a JPEG releases the GIL and parallelises well, while copying the
decoded image into a numpy array holds the GIL for its whole duration. The store
runs that copy on the consuming thread for exactly that reason, but it is still
serial work, and it is what bends the efficiency column down as the threads are
added.

## Summary

| | |
|---|---|
| Single-threaded | {single.seconds:.3f}s ({count / single.seconds:.0f} images/s) |
| Fastest | {best.num_workers} workers, {best.seconds:.3f}s ({count / best.seconds:.0f} images/s) |
| Speed-up | {_format_ratio(single.seconds, best.seconds)} |

## Measurements

{_format_table(measurements, count)}

## Environment

| | |
|---|---|
| pyvisim | {_package_version("pyvisim")} |
| Python | {platform.python_version()} |
| NumPy | {_package_version("numpy")} |
| Pillow | {_package_version("pillow")} |
| Platform | {platform.platform()} |
| CPUs | {os.cpu_count()} |
| Prefetch window | {prefetch} images (batch size {_BATCH_SIZE} x {_NUM_PREFETCH_BATCHES} batches) |
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure how the gallery reading of "
        "InMemoryImageEmbeddingStore scales with its number of worker threads."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "benchmarks" / "num_workers.md",
        help="Markdown file the report is written to.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of gallery images read per run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(_WORKER_COUNTS),
        help="Worker counts to measure. The first one is the baseline.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=_ROUNDS,
        help="Times every worker count is measured.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_BATCH_SIZE,
        help="Embedder batch size the prefetch window is derived from.",
    )
    parser.add_argument(
        "--num-prefetch-batches",
        type=int,
        default=_NUM_PREFETCH_BATCHES,
        help="Batches the reading threads may run ahead.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the image sampling."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure every worker count and write the report."""
    args = _parse_args(argv)
    prefetch = args.batch_size * args.num_prefetch_batches
    paths = _gallery_paths(args.num_images, args.seed)

    _log(f"Warming the page cache with {len(paths)} images.")
    _warm_page_cache(paths)

    measurements = _measure(paths, args.workers, args.rounds, prefetch)
    report = _format_report(measurements, len(paths), args.rounds, prefetch, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    _log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
