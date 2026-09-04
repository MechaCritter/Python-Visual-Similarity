"""
Measure what the batch size buys every pyvisim similarity metric.

Every metric processes the same batch of Oxford Flower images once per batch
size, and the wall-clock time and the peak resident memory of each run are
written to a markdown report. The GPU is disabled so that a single memory
number (the resident set size of this process) describes every metric.

Run it with::

    uv run --extra nn --group bench python scripts/benchmark_batch_size.py
"""

import argparse
import ctypes
import gc
import os
import platform
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import cast

import numpy as np
import psutil
import torch
from PIL import Image

from pyvisim._base_classes import ImageEmbedderBase, SimilarityMetric
from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder
from pyvisim.datasets import OxfordFlowerDataset
from pyvisim.features import DeepConvFeature, RootSIFT
from pyvisim.neural_networks import (
    BCESiameseNetwork,
    ClipEmbedder,
    ContrastiveSiameseNetwork,
    TripletNeuralNetwork,
)
from pyvisim.pixelwise import PSNR
from pyvisim.structural import MSSSIM, SSIM
from pyvisim.typing import FloatNumpyArray, UInt8NumpyArray

_SCRIPT_REF = "scripts/benchmark_batch_size.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Batch sizes every metric is measured at, in the order they are run.
_BATCH_SIZES = (1, 4, 16, 32)

_DATASET_SPLIT = "train"
#: Square side length every image is resized to. The pixel-wise and structural
#: metrics compare aligned grids, so the batch has to hold one single shape.
_IMAGE_SIZE = 256
#: Images of one untimed warm-up call, which pays the one-off costs (lazy
#: imports, kernel compilation, allocator growth) outside of the measurements.
_WARMUP_IMAGES = 8
#: Images the K-Means / GMM vocabularies of the classic embedders are fitted on.
_VOCABULARY_IMAGES = 30
_N_CLUSTERS = 32
_N_COMPONENTS = 16

#: Fixed thread budget of every run.
_NUM_THREADS = 4
#: How often the resident set size is sampled during a timed call, in seconds.
_RSS_SAMPLING_INTERVAL = 0.002

#: One batch of canonical ``uint8`` images.
_Images = list[UInt8NumpyArray]
#: Builds the metric under test from the vocabulary images.
_Builder = Callable[[_Images], SimilarityMetric]
#: Runs the workload of one metric over a batch of images.
_Runner = Callable[[SimilarityMetric, _Images], FloatNumpyArray]


def _log(message: str) -> None:
    """Print a progress line right away, so a long run can be followed."""
    print(message, flush=True)


def _set_batch_size(metric: SimilarityMetric, batch_size: int) -> None:
    """Set the batch size of a single metric."""
    metric.set_batch_size(batch_size)


def _set_pipeline_batch_size(metric: SimilarityMetric, batch_size: int) -> None:
    """Set the batch size of a pipeline and of every embedder inside it."""
    pipeline = cast(Pipeline, metric)
    pipeline.set_batch_size(batch_size)
    for embedder in pipeline.embedders:
        embedder.set_batch_size(batch_size)


@dataclass(frozen=True)
class Workload:
    """
    One metric together with the call that is timed.

    :param label: Name of the metric in the report.
    :param configuration: How the metric is configured and what one call does.
    :param build: Builds the metric from the vocabulary images.
    :param run: Runs the timed call.
    :param apply_batch_size: Applies a batch size to the built metric.
    """

    label: str
    configuration: str
    build: _Builder
    run: _Runner
    apply_batch_size: Callable[[SimilarityMetric, int], None] = _set_batch_size


@dataclass(frozen=True)
class Measurement:
    """
    What one batch size cost.

    :param batch_size: Batch size the call ran with.
    :param seconds: Wall-clock duration of the call.
    :param peak_rss_mib: Peak resident memory above the pre-call baseline.
    :param max_deviation: Largest absolute difference between this call's
        result and the ``batch_size=1`` result.
    """

    batch_size: int
    seconds: float
    peak_rss_mib: float
    max_deviation: float


@dataclass(frozen=True)
class Result:
    """
    Everything measured for one metric.

    :param label: Name of the metric in the report.
    :param configuration: How the metric is configured and what one call does.
    :param measurements: One measurement per batch size, empty when the metric
        could not be measured.
    :param error: Why the metric could not be measured, if it could not.
    """

    label: str
    configuration: str
    measurements: list[Measurement]
    error: str | None = None


class PeakRssProbe:
    """
    Samples the resident set size of this process in a background thread.

    The probe records the baseline when it is created and the peak while it is
    running, so :attr:`peak_mib` reports the memory one call needed on top of
    what the process already held.

    :param interval: Seconds between two samples.
    """

    def __init__(self, interval: float = _RSS_SAMPLING_INTERVAL) -> None:
        self._interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._baseline = self._rss()
        self._peak = self._baseline

    def _rss(self) -> int:
        """Return the current resident set size of this process, in bytes."""
        return int(self._process.memory_info().rss)

    def _sample(self) -> None:
        """Track the highest resident set size until the probe is stopped."""
        while not self._stop.is_set():
            self._peak = max(self._peak, self._rss())
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakRssProbe":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._rss())

    @property
    def peak_mib(self) -> float:
        """Peak resident memory above the baseline, in MiB."""
        return (self._peak - self._baseline) / 1024**2


def _release_free_memory() -> None:
    """
    Give the heap freed by the previous run back to the operating system.

    Without it the allocator keeps the pages of the previous run, and the next
    run allocates inside them instead of growing the resident set, which would
    report a peak far below what the call actually needed. The trim is a glibc
    extension: elsewhere the garbage collection alone has to do.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _load_images(
    count: int, size: int, seed: int
) -> tuple[_Images, UInt8NumpyArray, _Images]:
    """
    Draw the benchmark images from the Oxford Flower dataset.

    :param count: Number of images of the timed batch.
    :param size: Square side length every image is resized to.
    :param seed: Seed of the image sampling.
    :return: The timed batch, the reference image the pair-wise metrics score
        it against, and the vocabulary images of the classic embedders.
    """
    dataset = OxfordFlowerDataset(purpose=_DATASET_SPLIT)
    total = count + 1 + _VOCABULARY_IMAGES
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=total, replace=False)

    def resized(index: int) -> UInt8NumpyArray:
        image, _, _ = dataset[int(index)]
        square = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
        return cast(UInt8NumpyArray, np.asarray(square))

    images = [resized(index) for index in indices]
    return images[:count], images[count], images[count + 1 :]


def _embed(metric: SimilarityMetric, images: _Images) -> FloatNumpyArray:
    """Embed a batch of images."""
    return cast(ImageEmbedderBase, metric).embed(images)


def _score_against(reference: UInt8NumpyArray) -> _Runner:
    """Build a runner scoring every image of a batch against one reference image."""

    def run(metric: SimilarityMetric, images: _Images) -> FloatNumpyArray:
        return metric.similarity_score(images, reference)

    return run


def _learn(
    embedder: VLADEmbedder | FisherVectorEmbedder, images: _Images
) -> VLADEmbedder | FisherVectorEmbedder:
    """Learn the visual vocabulary of a classic embedder and return it."""
    embedder.learn(images)
    return embedder


def _build_vlad(images: _Images) -> SimilarityMetric:
    """Build a RootSIFT VLAD embedder with a fitted vocabulary."""
    return _learn(
        VLADEmbedder(feature_extractor=RootSIFT(), n_clusters=_N_CLUSTERS), images
    )


def _build_deep_vlad(images: _Images) -> SimilarityMetric:
    """Build a VLAD embedder over deep convolutional features."""
    return _learn(
        VLADEmbedder(feature_extractor=DeepConvFeature(), n_clusters=_N_CLUSTERS),
        images,
    )


def _build_fisher(images: _Images) -> SimilarityMetric:
    """Build a RootSIFT Fisher Vector embedder with a fitted vocabulary."""
    return _learn(
        FisherVectorEmbedder(feature_extractor=RootSIFT(), n_components=_N_COMPONENTS),
        images,
    )


def _build_pipeline(images: _Images) -> SimilarityMetric:
    """Build a pipeline joining a VLAD and a Fisher Vector embedder."""
    return Pipeline(
        [
            cast(VLADEmbedder, _build_vlad(images)),
            cast(FisherVectorEmbedder, _build_fisher(images)),
        ]
    )


def _workloads(reference: UInt8NumpyArray) -> list[Workload]:
    """
    Describe every metric under test.

    :param reference: Image the pair-wise metrics score the batch against.
    :return: One workload per metric, in report order.
    """
    score = _score_against(reference)
    pairs = "scores the batch against one reference image"
    embeds = "embeds the batch"
    return [
        Workload(
            "PSNR",
            f"`PSNR()`, {pairs}.",
            lambda _: PSNR(),
            score,
        ),
        Workload(
            "SSIM",
            f"`SSIM(num_workers={_NUM_THREADS})`, {pairs}.",
            lambda _: SSIM(num_workers=_NUM_THREADS),
            score,
        ),
        Workload(
            "MS-SSIM",
            f"`MSSSIM(num_workers={_NUM_THREADS})`, {pairs}.",
            lambda _: MSSSIM(num_workers=_NUM_THREADS),
            score,
        ),
        Workload(
            "VLAD (RootSIFT)",
            f"`VLADEmbedder(RootSIFT(), n_clusters={_N_CLUSTERS})`, {embeds}.",
            _build_vlad,
            _embed,
        ),
        Workload(
            "VLAD (DeepConvFeature)",
            f"`VLADEmbedder(DeepConvFeature(), n_clusters={_N_CLUSTERS})`, "
            f"{embeds}. The extractor pushes a whole batch through VGG16 in one "
            "forward pass.",
            _build_deep_vlad,
            _embed,
        ),
        Workload(
            "Fisher Vector (RootSIFT)",
            f"`FisherVectorEmbedder(RootSIFT(), n_components={_N_COMPONENTS})`, "
            f"{embeds}.",
            _build_fisher,
            _embed,
        ),
        Workload(
            "Pipeline (VLAD + Fisher Vector)",
            f"`Pipeline([VLADEmbedder, FisherVectorEmbedder])` over RootSIFT, "
            f"{embeds}. The batch size is applied to the pipeline and to both "
            "embedders inside it.",
            _build_pipeline,
            _embed,
            _set_pipeline_batch_size,
        ),
        Workload(
            "CLIP",
            f'`ClipEmbedder("ViT-B-32", "openai")`, {embeds}.',
            lambda _: ClipEmbedder(),
            _embed,
        ),
        Workload(
            "Contrastive Siamese network",
            f'`ContrastiveSiameseNetwork("resnet18")`, {embeds}.',
            lambda _: ContrastiveSiameseNetwork(),
            _embed,
        ),
        Workload(
            "Triplet network",
            f'`TripletNeuralNetwork("resnet18")`, {embeds}.',
            lambda _: TripletNeuralNetwork(),
            _embed,
        ),
        Workload(
            "BCE Siamese network",
            f'`BCESiameseNetwork("resnet18")`, {pairs}.',
            lambda _: BCESiameseNetwork(),
            score,
        ),
    ]


def _max_deviation(reference: FloatNumpyArray | None, result: FloatNumpyArray) -> float:
    """
    Compare a result against the one the ``batch_size=1`` run produced.

    :param reference: Result of the ``batch_size=1`` run, or ``None`` for that
        run itself.
    :param result: Result of the current run.
    :return: The largest absolute difference between the two, ignoring the
        entries that are not finite in both.
    """
    if reference is None:
        return 0.0
    difference = np.abs(
        np.asarray(result, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    )
    finite = difference[np.isfinite(difference)]
    return float(finite.max()) if finite.size else 0.0


def _measure(
    workload: Workload,
    metric: SimilarityMetric,
    images: _Images,
    batch_size: int,
    baseline: FloatNumpyArray | None,
) -> tuple[Measurement, FloatNumpyArray]:
    """
    Time one call of a workload and record how much memory it needed.

    :param workload: Workload under test.
    :param metric: The built metric.
    :param images: The timed batch of images.
    :param batch_size: Batch size the call runs with.
    :param baseline: Result of the ``batch_size=1`` run, if it already ran.
    :return: The measurement and the result of the call.
    """
    workload.apply_batch_size(metric, batch_size)
    _release_free_memory()
    with PeakRssProbe() as probe:
        start = time.perf_counter()
        result = workload.run(metric, images)
        seconds = time.perf_counter() - start
    measurement = Measurement(
        batch_size=batch_size,
        seconds=seconds,
        peak_rss_mib=probe.peak_mib,
        max_deviation=_max_deviation(baseline, result),
    )
    return measurement, result


def _run_workload(workload: Workload, images: _Images, vocabulary: _Images) -> Result:
    """
    Build one metric and measure it at every batch size.

    A metric that cannot be built or run (e.g. because its weights cannot be
    fetched) is reported with the reason instead of aborting the whole run.

    :param workload: Workload under test.
    :param images: The timed batch of images.
    :param vocabulary: Images the classic embedders fit their vocabulary on.
    :return: The measurements of the metric, or the error that stopped them.
    """
    _log(f"[{workload.label}] building")
    try:
        metric = workload.build(vocabulary)
        workload.apply_batch_size(metric, _BATCH_SIZES[-1])
        workload.run(metric, images[:_WARMUP_IMAGES])
    except Exception as error:
        # A metric that cannot be built is reported, not raised: the other
        # metrics of the run are still worth measuring.
        _log(f"[{workload.label}] skipped: {error}")
        return Result(workload.label, workload.configuration, [], str(error))

    measurements: list[Measurement] = []
    baseline: FloatNumpyArray | None = None
    for batch_size in _BATCH_SIZES:
        try:
            measurement, result = _measure(
                workload, metric, images, batch_size, baseline
            )
        except Exception as error:
            _log(f"[{workload.label}] batch_size={batch_size} failed: {error}")
            return Result(
                workload.label, workload.configuration, measurements, str(error)
            )
        if baseline is None:
            baseline = result
        measurements.append(measurement)
        _log(
            f"[{workload.label}] batch_size={batch_size}: "
            f"{measurement.seconds:.2f} s, {measurement.peak_rss_mib:.1f} MiB"
        )
    return Result(workload.label, workload.configuration, measurements)


def _format_seconds(value: float) -> str:
    """Format a duration for the report tables, in seconds."""
    return f"{value:.3f}" if value < 10.0 else f"{value:.1f}"


def _format_ratio(value: float, baseline: float) -> str:
    """Format a ratio against a baseline, or ``n/a`` when there is nothing to divide by."""
    if baseline <= 0.0:
        return "n/a"
    return f"{value / baseline:.2f}x"


def _format_deviation(measurement: Measurement) -> str:
    """Format the deviation of one measurement from the ``batch_size=1`` result."""
    if measurement.batch_size == _BATCH_SIZES[0]:
        return "reference"
    return f"{measurement.max_deviation:.2e}"


def _format_detail_table(result: Result) -> str:
    """Render the per-batch-size measurements of one metric as a markdown table."""
    lines = [
        "| Batch size | Time (s) | Speed-up | Peak RSS (MiB) | Memory vs. batch 1 | Max. deviation |",
        "|---|---|---|---|---|---|",
    ]
    unbatched = result.measurements[0]
    for measurement in result.measurements:
        lines.append(
            f"| {measurement.batch_size} | {_format_seconds(measurement.seconds)} "
            f"| {_format_ratio(unbatched.seconds, measurement.seconds)} "
            f"| {measurement.peak_rss_mib:.1f} "
            f"| {_format_ratio(measurement.peak_rss_mib, unbatched.peak_rss_mib)} "
            f"| {_format_deviation(measurement)} |"
        )
    return "\n".join(lines)


def _best(result: Result) -> Measurement:
    """Return the fastest measurement of one metric."""
    return min(result.measurements, key=lambda measurement: measurement.seconds)


def _format_summary_table(results: Sequence[Result]) -> str:
    """Render the headline numbers of every metric as a markdown table."""
    lines = [
        "| Metric | Batch 1 (s) | Fastest batch | Speed-up | Peak RSS (MiB) | Memory vs. batch 1 |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        if not result.measurements:
            lines.append(f"| {result.label} | not measured | | | | |")
            continue
        unbatched, best = result.measurements[0], _best(result)
        lines.append(
            f"| {result.label} | {_format_seconds(unbatched.seconds)} | {best.batch_size} "
            f"| {_format_ratio(unbatched.seconds, best.seconds)} "
            f"| {best.peak_rss_mib:.1f} "
            f"| {_format_ratio(best.peak_rss_mib, unbatched.peak_rss_mib)} |"
        )
    return "\n".join(lines)


def _format_metric_section(result: Result) -> str:
    """Render the full markdown subsection of one metric."""
    body = (
        _format_detail_table(result)
        if result.measurements
        else "This metric could not be measured."
    )
    failure = f"\n\n> Failed with: `{result.error}`" if result.error else ""
    return f"""### {result.label}

{result.configuration}

{body}{failure}
"""


def _package_version(name: str) -> str:
    """Resolve an installed package version, or ``"unknown"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_report(results: Sequence[Result], count: int, size: int, seed: int) -> str:
    """Render the complete markdown report."""
    sections = "\n".join(_format_metric_section(result) for result in results)
    batch_sizes = ", ".join(str(batch_size) for batch_size in _BATCH_SIZES)
    return f"""# Batch size benchmark

> [!IMPORTANT]
> This file was generated by the script [`{_SCRIPT_REF}`](../../{_SCRIPT_REF}).
> **Do not edit manually!**

Every metric processes the same {count} images of the Oxford Flower dataset
(`{_DATASET_SPLIT}` split, seed {seed}), resized to {size}x{size} RGB, once per
batch size ({batch_sizes}). One untimed warm-up call over {_WARMUP_IMAGES}
images precedes the measurements, so the one-off costs of the first call are
not counted.

The **speed-up** is the wall-clock time of the `batch_size=1` run divided by
the one of the run, and **memory** is the peak resident set size of the run
divided by the one of the `batch_size=1` run. The peak is sampled every
{_RSS_SAMPLING_INTERVAL * 1000:.0f} ms and is measured above the memory the
process already held, which the input batch is part of: it is the memory the
call itself needed, not the memory of the whole process. The heap is trimmed
between two runs, but a resident set size is a noisy quantity, so read the
memory columns as an order of magnitude rather than as an exact figure.

**Max. deviation** is the largest absolute difference between the scores of the
run and the scores of the `batch_size=1` run. It is not zero everywhere:
splitting a batch changes the order the floating point reductions run in.

## Summary

{_format_summary_table(results)}

## Environment

| | |
|---|---|
| pyvisim | {_package_version("pyvisim")} |
| Python | {platform.python_version()} |
| NumPy | {_package_version("numpy")} |
| PyTorch | {_package_version("torch")} ({torch.get_num_threads()} threads, CPU only) |
| Platform | {platform.platform()} |
| Threads | `PYVISIM_NUM_THREADS={_NUM_THREADS}` |

## Metrics

{sections}"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure the speed-up and the memory cost of the batch "
        "size of every pyvisim similarity metric."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "benchmarks" / "batch_size.md",
        help="Markdown file the report is written to.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Number of images every metric processes per call.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=_IMAGE_SIZE,
        help="Square side length every image is resized to.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the image sampling."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure every metric and write the report."""
    args = _parse_args(argv)
    os.environ["PYVISIM_NUM_THREADS"] = str(_NUM_THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    if torch.cuda.is_available():
        raise RuntimeError(
            "The GPU is still visible. Its memory is not part of the resident "
            "set size this benchmark measures."
        )
    torch.set_num_threads(_NUM_THREADS)
    images, reference, vocabulary = _load_images(
        args.num_images, args.image_size, args.seed
    )
    results = [
        _run_workload(workload, images, vocabulary)
        for workload in _workloads(reference)
    ]
    report = _format_report(results, len(images), args.image_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    _log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
