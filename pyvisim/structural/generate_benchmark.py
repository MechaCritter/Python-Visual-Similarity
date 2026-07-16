"""
Generate the "Benchmark" documentation section for ``pyvisim.structural``.

The script benchmarks :class:`pyvisim.structural.SSIM` and
:class:`pyvisim.structural.MSSSIM` against well-known reference ("baseline")
implementations, mirroring ``notebooks/benchmark_structural.ipynb``:

- ``SSIM`` baseline: ``skimage.metrics.structural_similarity`` configured for
  the Wang et al. (2004) reference implementation (11x11 Gaussian window,
  ``sigma=1.5``, population statistics, ``data_range=255``).
- ``MSSSIM`` baseline: ``torchmetrics``
  ``multiscale_structural_similarity_index_measure`` (11x11 Gaussian window,
  ``sigma=1.5``, paper weights, ``normalize="relu"``).

All images are sampled without replacement from the Oxford Flower dataset
(:class:`pyvisim.datasets.OxfordFlowerDataset`), with a *disjoint* image
subset per experiment and per metric. Two experiment families run per metric:

- **Accuracy**: each sampled image (native resolution) is scored against a
  noise-distorted copy of itself by pyvisim and by the baseline; the absolute
  score difference is recorded per image.
- **Runtime**: wall-clock time of one full scoring call, repeated several
  times, across four workloads — one pair (512x512), one expanded pair (a
  flower image resized to 1024x1024), a 4x4 score matrix (16 pairs) and a
  16x16 score matrix (256 pairs) of 256x256 images. The thread budget is
  fixed: ``PYVISIM_NUM_THREADS=4`` and the metrics run with ``num_workers=2``.

Outputs, written to ``docs/structural/`` by default:

- ``benchmark/benchmark_results.json`` — per-image errors (with the baseline
  spelled out), raw runtimes and the full configuration.
- ``benchmark/{ssim,ms_ssim}_runtime_barplot.png`` — median pyvisim vs.
  baseline runtimes.
- The auto-generated ``## Benchmark`` section of ``README.md``, replaced
  in-place between the ``benchmark:begin`` / ``benchmark:end`` markers.

Usage::

    python -m pyvisim.structural.generate_benchmark [--seed N] [--repeats N]
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure as tm_msssim,
)

from ..base import DenseMetricBase
from ..datasets import OxfordFlowerDataset
from ..typing import UInt8NumpyArray
from ._ms_ssim import MSSSIM
from ._ssim import SSIM

_SCRIPT_REF = "pyvisim/structural/generate_benchmark.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixed thread budget of every run (see module docstring).
_NUM_THREADS = 4
_NUM_WORKERS = 2

_DATASET_SPLIT = "train"
_NOISE_STD = 15.0
_N_ACCURACY_IMAGES = 8
_SINGLE_PAIR_SIZE = 512
_EXPANDED_PAIR_SIZE = 1024
_BATCH_IMAGE_SIZE = 256
_SMALL_BATCH = 4
_LARGE_BATCH = 8

_SECTION_BEGIN = "<!-- benchmark:begin -->"
_SECTION_END = "<!-- benchmark:end -->"

# Chart chrome (light mode) and the two fixed series slots, following the
# validated default dataviz palette: slot 1 (blue) = pyvisim, slot 2 (green)
# = baseline.
_COLOR_PYVISIM = "#2a78d6"
_COLOR_BASELINE = "#008300"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"

#: One sampled image: (base file name, RGB uint8 array).
_NamedImage = tuple[str, UInt8NumpyArray]


@dataclass(frozen=True)
class ScenarioResult:
    """
    Raw runtimes of one workload.

    :param image_names: Base file names of the sampled dataset images.
    :param image_size: Side length the images were resized to, in pixels.
    :param n_pairs: Number of image pairs scored per call.
    :param pyvisim_ms: Wall-clock time of each pyvisim call, in milliseconds.
    :param baseline_ms: Wall-clock time of each baseline call, in
        milliseconds.
    """

    image_names: list[str]
    image_size: int
    n_pairs: int
    pyvisim_ms: list[float]
    baseline_ms: list[float]


@dataclass(frozen=True)
class MetricBenchmark:
    """
    Complete benchmark results of one structural metric.

    :param metric_label: Human-readable metric name ("SSIM" or "MS-SSIM").
    :param slug: File-name-safe identifier of the metric.
    :param baseline_label: Short baseline name used in plot legends.
    :param baseline: Full description of the baseline implementation,
        including the library version and every non-default argument.
    :param accuracy: Per-image scores keyed by base file name; each value
        holds the ``pyvisim`` score, the ``baseline`` score and their
        ``abs_error``.
    :param runtime: Per-workload raw runtimes keyed by scenario label.
    """

    metric_label: str
    slug: str
    baseline_label: str
    baseline: str
    accuracy: dict[str, dict[str, float]] = field(default_factory=dict)
    runtime: dict[str, ScenarioResult] = field(default_factory=dict)


def _package_version(name: str) -> str:
    """
    Resolve an installed package version.

    :param name: Distribution name.
    :return: The version string, or ``"unknown"`` if not installed.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _sample_disjoint_images(
    dataset: OxfordFlowerDataset,
    rng: np.random.Generator,
    counts: Sequence[int],
) -> list[list[_NamedImage]]:
    """
    Sample disjoint groups of dataset images without replacement.

    :param dataset: Source dataset.
    :param rng: Seeded random generator.
    :param counts: Number of images per group.
    :return: One list of (base file name, RGB image) tuples per count.
    :raises ValueError: If the dataset holds fewer images than requested.
    """
    total = sum(counts)
    if total > len(dataset):
        raise ValueError(
            f"Cannot sample {total} distinct images from a dataset of "
            f"{len(dataset)} images."
        )
    indices = rng.choice(len(dataset), size=total, replace=False)
    groups: list[list[_NamedImage]] = []
    start = 0
    for count in counts:
        group: list[_NamedImage] = []
        for index in indices[start : start + count]:
            image, _, path = dataset[int(index)]
            group.append((os.path.basename(path), image))
        groups.append(group)
        start += count
    return groups


def _resize(image: UInt8NumpyArray, size: int) -> UInt8NumpyArray:
    """
    Resize an RGB image to a square side length.

    :param image: ``(H, W, C)`` uint8 image.
    :param size: Target side length, in pixels.
    :return: The resized ``(size, size, C)`` uint8 image.
    """
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    return cast(UInt8NumpyArray, np.asarray(resized))


def _add_noise(
    image: UInt8NumpyArray, std: float, rng: np.random.Generator
) -> UInt8NumpyArray:
    """
    Add clipped Gaussian noise to a uint8 image.

    :param image: ``(H, W[, C])`` uint8 image.
    :param std: Standard deviation of the noise, in intensity levels.
    :param rng: Seeded random generator.
    :return: The distorted uint8 image.
    """
    noisy = image.astype(np.float64) + rng.normal(0.0, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _skimage_ssim(image_a: UInt8NumpyArray, image_b: UInt8NumpyArray) -> float:
    """
    Score one RGB pair with the scikit-image SSIM baseline.

    :param image_a: First ``(H, W, C)`` uint8 image.
    :param image_b: Second ``(H, W, C)`` uint8 image, same shape.
    :return: The mean SSIM score.
    """
    return float(
        structural_similarity(
            image_a,
            image_b,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=255,
            channel_axis=-1,
        )
    )


def _to_bchw(images: Iterable[UInt8NumpyArray], dtype: type) -> torch.Tensor:
    """
    Stack RGB images into a ``(B, C, H, W)`` torch tensor.

    :param images: Same-shape ``(H, W, C)`` uint8 images.
    :param dtype: NumPy scalar type of the tensor (e.g. ``np.float32``).
    :return: The stacked tensor.
    """
    stacked = np.stack(list(images)).astype(dtype).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def _torchmetrics_msssim(rows: torch.Tensor, cols: torch.Tensor) -> float:
    """
    Score aligned pairs with the torchmetrics MS-SSIM baseline.

    :param rows: ``(B, C, H, W)`` tensor, one image per pair.
    :param cols: ``(B, C, H, W)`` tensor, aligned with ``rows``.
    :return: The mean MS-SSIM score over the batch.
    """
    return float(
        tm_msssim(
            rows,
            cols,
            data_range=255.0,
            kernel_size=11,
            sigma=1.5,
            normalize="relu",
        )
    )


def _measure_accuracy(
    metric: DenseMetricBase,
    baseline_fn: Callable[[UInt8NumpyArray, UInt8NumpyArray], float],
    images: Sequence[_NamedImage],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """
    Score each image against a noise-distorted copy with both implementations.

    :param metric: pyvisim metric under test.
    :param baseline_fn: Baseline scoring one RGB uint8 pair.
    :param images: Named native-resolution images.
    :param rng: Seeded random generator for the distortion noise.
    :return: Per-image scores and absolute errors, keyed by base file name.
    """
    results: dict[str, dict[str, float]] = {}
    for name, image in images:
        distorted = _add_noise(image, _NOISE_STD, rng)
        ours = float(metric.similarity_score(image, distorted)[0, 0])
        reference = baseline_fn(image, distorted)
        results[name] = {
            "pyvisim": ours,
            "baseline": reference,
            "abs_error": abs(ours - reference),
        }
    return results


def _time_ms(fn: Callable[[], object], repeats: int) -> list[float]:
    """
    Measure the wall-clock time of repeated calls, after one warm-up call.

    :param fn: Zero-argument callable to time.
    :param repeats: Number of timed calls.
    :return: One duration per call, in milliseconds.
    """
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(1000.0 * (time.perf_counter() - start))
    return times


def _build_galleries(
    groups: Sequence[list[_NamedImage]], rng: np.random.Generator
) -> dict[str, tuple[list[UInt8NumpyArray], list[UInt8NumpyArray], list[str], int]]:
    """
    Turn the four sampled runtime image groups into scoring workloads.

    :param groups: Sampled image groups for, in order: the single pair, the
        expanded pair, the batch of 4 and the batch of 16.
    :param rng: Seeded random generator for the expanded pair's distortion.
    :return: Per-scenario ``(gallery_a, gallery_b, image_names, image_size)``,
        where every image of ``gallery_a`` is scored against every image of
        ``gallery_b``.
    """
    single_group, expanded_group, small_group, large_group = groups
    single = [_resize(image, _SINGLE_PAIR_SIZE) for _, image in single_group]
    expanded = _resize(expanded_group[0][1], _EXPANDED_PAIR_SIZE)
    small = [_resize(image, _BATCH_IMAGE_SIZE) for _, image in small_group]
    large = [_resize(image, _BATCH_IMAGE_SIZE) for _, image in large_group]
    return {
        f"1 pair ({_SINGLE_PAIR_SIZE}x{_SINGLE_PAIR_SIZE})": (
            [single[0]],
            [single[1]],
            [name for name, _ in single_group],
            _SINGLE_PAIR_SIZE,
        ),
        f"1 expanded pair ({_EXPANDED_PAIR_SIZE}x{_EXPANDED_PAIR_SIZE})": (
            [expanded],
            [_add_noise(expanded, _NOISE_STD, rng)],
            [name for name, _ in expanded_group],
            _EXPANDED_PAIR_SIZE,
        ),
        f"batch of {_SMALL_BATCH} ({_SMALL_BATCH**2} pairs)": (
            small,
            small,
            [name for name, _ in small_group],
            _BATCH_IMAGE_SIZE,
        ),
        f"batch of {_LARGE_BATCH} ({_LARGE_BATCH**2} pairs)": (
            large,
            large,
            [name for name, _ in large_group],
            _BATCH_IMAGE_SIZE,
        ),
    }


def _time_pyvisim_matrix(
    metric: DenseMetricBase,
    gallery_a: list[UInt8NumpyArray],
    gallery_b: list[UInt8NumpyArray],
    repeats: int,
) -> list[float]:
    """
    Time full pyvisim score-matrix calls.

    :param metric: pyvisim metric under test.
    :param gallery_a: First image gallery.
    :param gallery_b: Second image gallery.
    :param repeats: Number of timed calls.
    :return: One duration per call, in milliseconds.
    """
    return _time_ms(lambda: metric.similarity_score(gallery_a, gallery_b), repeats)


def _time_skimage_matrix(
    gallery_a: list[UInt8NumpyArray],
    gallery_b: list[UInt8NumpyArray],
    repeats: int,
) -> list[float]:
    """
    Time the scikit-image baseline over the same pairs, via a Python loop.

    :param gallery_a: First image gallery.
    :param gallery_b: Second image gallery.
    :param repeats: Number of timed calls.
    :return: One duration per call, in milliseconds.
    """

    def score_all() -> list[float]:
        return [_skimage_ssim(a, b) for a in gallery_a for b in gallery_b]

    return _time_ms(score_all, repeats)


def _time_torchmetrics_matrix(
    gallery_a: list[UInt8NumpyArray],
    gallery_b: list[UInt8NumpyArray],
    repeats: int,
) -> list[float]:
    """
    Time the torchmetrics baseline over the same pairs, as one stacked batch.

    The tensor stacking is done once outside the timed region, mirroring the
    benchmark notebook (pyvisim timings, by contrast, always include the full
    input pipeline).

    :param gallery_a: First image gallery.
    :param gallery_b: Second image gallery.
    :param repeats: Number of timed calls.
    :return: One duration per call, in milliseconds.
    """
    stack_a = _to_bchw(gallery_a, np.float32)
    stack_b = _to_bchw(gallery_b, np.float32)
    rows = stack_a.repeat_interleave(stack_b.shape[0], dim=0)
    cols = stack_b.repeat(stack_a.shape[0], 1, 1, 1)
    return _time_ms(lambda: _torchmetrics_msssim(rows, cols), repeats)


def _measure_runtime(
    metric: DenseMetricBase,
    baseline_timer: Callable[
        [list[UInt8NumpyArray], list[UInt8NumpyArray], int], list[float]
    ],
    groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> dict[str, ScenarioResult]:
    """
    Time every runtime workload for one metric and its baseline.

    :param metric: pyvisim metric under test.
    :param baseline_timer: Baseline timing function for one workload.
    :param groups: Sampled image groups (see :func:`_build_galleries`).
    :param rng: Seeded random generator.
    :param repeats: Number of timed calls per workload.
    :return: Raw runtimes keyed by scenario label.
    """
    results: dict[str, ScenarioResult] = {}
    for label, (gallery_a, gallery_b, names, size) in _build_galleries(
        groups, rng
    ).items():
        results[label] = ScenarioResult(
            image_names=names,
            image_size=size,
            n_pairs=len(gallery_a) * len(gallery_b),
            pyvisim_ms=_time_pyvisim_matrix(metric, gallery_a, gallery_b, repeats),
            baseline_ms=baseline_timer(gallery_a, gallery_b, repeats),
        )
    return results


def _format_ms(value: float) -> str:
    """
    Format a duration in milliseconds for labels and tables.

    :param value: Duration in milliseconds.
    :return: The formatted duration.
    """
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def _wrap_label(label: str) -> str:
    """
    Break a scenario label before its parenthesis for axis ticks.

    :param label: Scenario label.
    :return: The two-line tick label.
    """
    return label.replace(" (", "\n(")


def _style_axes(ax: object) -> None:
    """
    Apply the shared recessive chart chrome to a matplotlib axes.

    :param ax: The ``matplotlib.axes.Axes`` to style.
    """
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_runtime_barplot(
    runtime: dict[str, ScenarioResult],
    baseline_label: str,
    title: str,
    path: Path,
) -> None:
    """
    Plot the median pyvisim vs. baseline runtime per scenario as grouped bars.

    :param runtime: Raw runtimes keyed by scenario label.
    :param baseline_label: Legend label of the baseline series.
    :param title: Figure title.
    :param path: Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    positions = np.arange(len(runtime))
    medians = {
        "pyvisim": [float(np.median(r.pyvisim_ms)) for r in runtime.values()],
        baseline_label: [float(np.median(r.baseline_ms)) for r in runtime.values()],
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=_SURFACE)
    _style_axes(ax)
    width = 0.38
    for (label, values), color, offset in zip(
        medians.items(), (_COLOR_PYVISIM, _COLOR_BASELINE), (-0.5, 0.5), strict=True
    ):
        bars = ax.bar(
            positions + offset * width, values, width=width, color=color, label=label
        )
        ax.bar_label(
            bars,
            labels=[_format_ms(value) for value in values],
            color=_INK_SECONDARY,
            fontsize=8,
            padding=2,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label) for label in runtime])
    ax.set_ylabel("median runtime per call (ms)", color=_INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=_INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _format_accuracy_table(accuracy: dict[str, dict[str, float]]) -> str:
    """
    Render the per-image accuracy results as a markdown table.

    :param accuracy: Per-image scores keyed by base file name.
    :return: The markdown table.
    """
    lines = [
        "| Image | pyvisim | Baseline | Abs. error |",
        "|---|---|---|---|",
    ]
    for name in sorted(accuracy):
        scores = accuracy[name]
        lines.append(
            f"| `{name}` | {scores['pyvisim']:.6f} | {scores['baseline']:.6f} "
            f"| {scores['abs_error']:.2e} |"
        )
    return "\n".join(lines)


def _format_runtime_table(runtime: dict[str, ScenarioResult]) -> str:
    """
    Render the per-scenario median runtimes as a markdown table.

    :param runtime: Raw runtimes keyed by scenario label.
    :return: The markdown table.
    """
    lines = [
        "| Scenario | Images | pyvisim (ms) | Baseline (ms) | Speed-up |",
        "|---|---|---|---|---|",
    ]
    for label, result in runtime.items():
        ours = float(np.median(result.pyvisim_ms))
        reference = float(np.median(result.baseline_ms))
        names = ", ".join(f"`{name}`" for name in result.image_names)
        lines.append(
            f"| {label}, {result.image_size}x{result.image_size} | {names} "
            f"| {_format_ms(ours)} | {_format_ms(reference)} "
            f"| {reference / ours:.1f}x |"
        )
    return "\n".join(lines)


def _format_metric_section(bench: MetricBenchmark) -> str:
    """
    Render the full markdown subsection of one metric.

    :param bench: Benchmark results of the metric.
    :return: The markdown subsection.
    """
    return f"""### {bench.metric_label}

Baseline: **{bench.baseline}**.

#### Accuracy

Each image (native resolution) is scored against a copy distorted with
Gaussian noise (std {_NOISE_STD:g}); the error is the absolute difference
between the pyvisim score and the baseline score.

{_format_accuracy_table(bench.accuracy)}

#### Runtime

Median wall-clock time per full scoring call ({bench.metric_label} of every
image in the first gallery against every image in the second). pyvisim
timings include the whole input pipeline (canonicalization to uint8,
stacking, scoring).

{_format_runtime_table(bench.runtime)}

![{bench.metric_label} median runtime](benchmark/{bench.slug}_runtime_barplot.png)
"""


def _format_section(
    benchmarks: Sequence[MetricBenchmark], seed: int, repeats: int
) -> str:
    """
    Render the complete auto-generated "Benchmark" markdown section.

    :param benchmarks: Results of every benchmarked metric.
    :param seed: Random seed the images were sampled with.
    :param repeats: Number of timed calls per runtime workload.
    :return: The markdown section, including the begin/end markers.
    """
    metric_sections = "\n".join(_format_metric_section(b) for b in benchmarks)
    return f"""{_SECTION_BEGIN}
## Benchmark

> [!IMPORTANT]
> **Do not edit this section manually.** It was generated by
> [`{_SCRIPT_REF}`](../../{_SCRIPT_REF}) (see
> [`benchmark/benchmark_results.json`](benchmark/benchmark_results.json) for
> the raw data); re-run
> `python -m pyvisim.structural.generate_benchmark` to refresh it.

All images are drawn without replacement from the Oxford Flower dataset
(`{_DATASET_SPLIT}` split, seed {seed}), with a disjoint image subset per
experiment and per metric. Thread budget: `PYVISIM_NUM_THREADS={_NUM_THREADS}`
with `num_workers={_NUM_WORKERS}`; every runtime is the wall-clock time of
one full scoring call ({repeats} timed calls after one warm-up).

{metric_sections}
{_SECTION_END}"""


def _inject_section(readme_path: Path, section: str) -> None:
    """
    Replace the auto-generated section of the README, in place.

    The section is delimited by the ``benchmark:begin`` / ``benchmark:end``
    markers. If the README does not exist yet, a stub holding only the
    section is created; if it exists without markers, the section is
    appended.

    :param readme_path: Path to the target markdown file.
    :param section: The full replacement section, including markers.
    :raises ValueError: If the README holds mismatched section markers.
    """
    if not readme_path.exists():
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(f"# Structural metrics\n\n{section}\n")
        return
    text = readme_path.read_text()
    has_begin, has_end = _SECTION_BEGIN in text, _SECTION_END in text
    if has_begin != has_end:
        raise ValueError(
            f"{readme_path} holds mismatched {_SECTION_BEGIN} / {_SECTION_END} markers."
        )
    if has_begin:
        before, _, rest = text.partition(_SECTION_BEGIN)
        _, _, after = rest.partition(_SECTION_END)
        text = f"{before}{section}{after}"
    else:
        text = f"{text.rstrip()}\n\n{section}\n"
    readme_path.write_text(text)


def _benchmark_ssim(
    accuracy_images: list[_NamedImage],
    runtime_groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> MetricBenchmark:
    """
    Benchmark :class:`SSIM` against scikit-image.

    :param accuracy_images: Native-resolution images of the accuracy run.
    :param runtime_groups: Image groups of the four runtime workloads.
    :param rng: Seeded random generator.
    :param repeats: Number of timed calls per runtime workload.
    :return: The complete SSIM benchmark results.
    """
    metric = SSIM(num_workers=_NUM_WORKERS)
    return MetricBenchmark(
        metric_label="SSIM",
        slug="ssim",
        baseline_label="scikit-image",
        baseline=(
            f"scikit-image {_package_version('scikit-image')} — "
            "skimage.metrics.structural_similarity(win_size=11, "
            "gaussian_weights=True, sigma=1.5, use_sample_covariance=False, "
            "data_range=255)"
        ),
        accuracy=_measure_accuracy(metric, _skimage_ssim, accuracy_images, rng),
        runtime=_measure_runtime(
            metric, _time_skimage_matrix, runtime_groups, rng, repeats
        ),
    )


def _benchmark_msssim(
    accuracy_images: list[_NamedImage],
    runtime_groups: Sequence[list[_NamedImage]],
    rng: np.random.Generator,
    repeats: int,
) -> MetricBenchmark:
    """
    Benchmark :class:`MSSSIM` against torchmetrics.

    :param accuracy_images: Native-resolution images of the accuracy run.
    :param runtime_groups: Image groups of the four runtime workloads.
    :param rng: Seeded random generator.
    :param repeats: Number of timed calls per runtime workload.
    :return: The complete MS-SSIM benchmark results.
    """

    def baseline_pair(image_a: UInt8NumpyArray, image_b: UInt8NumpyArray) -> float:
        return _torchmetrics_msssim(
            _to_bchw([image_a], np.float64), _to_bchw([image_b], np.float64)
        )

    metric = MSSSIM(num_workers=_NUM_WORKERS)
    return MetricBenchmark(
        metric_label="MS-SSIM",
        slug="ms_ssim",
        baseline_label="torchmetrics",
        baseline=(
            f"torchmetrics {_package_version('torchmetrics')} — "
            "torchmetrics.functional.image."
            "multiscale_structural_similarity_index_measure(data_range=255.0, "
            "kernel_size=11, sigma=1.5, normalize='relu')"
        ),
        accuracy=_measure_accuracy(metric, baseline_pair, accuracy_images, rng),
        runtime=_measure_runtime(
            metric, _time_torchmetrics_matrix, runtime_groups, rng, repeats
        ),
    )


def _write_outputs(
    benchmarks: Sequence[MetricBenchmark],
    output_dir: Path,
    readme_path: Path,
    seed: int,
    repeats: int,
) -> None:
    """
    Write the results JSON, the four plots and the README section.

    :param benchmarks: Results of every benchmarked metric.
    :param output_dir: Directory of the JSON and PNG outputs.
    :param readme_path: Markdown file receiving the "Benchmark" section.
    :param seed: Random seed the images were sampled with.
    :param repeats: Number of timed calls per runtime workload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "_generated_by": (
            f"{_SCRIPT_REF} — do not edit manually, re-run the script instead."
        ),
        "config": {
            "dataset": f"OxfordFlowerDataset (split: {_DATASET_SPLIT})",
            "seed": seed,
            "repeats": repeats,
            "noise_std": _NOISE_STD,
            "pyvisim_num_threads": _NUM_THREADS,
            "num_workers": _NUM_WORKERS,
            "versions": {
                name: _package_version(name)
                for name in (
                    "pyvisim",
                    "numpy",
                    "scikit-image",
                    "torch",
                    "torchmetrics",
                )
            },
        },
    }
    for bench in benchmarks:
        payload[bench.slug] = {
            "baseline": bench.baseline,
            "accuracy": bench.accuracy,
            "runtime": {label: asdict(r) for label, r in bench.runtime.items()},
        }
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    for bench in benchmarks:
        _plot_runtime_barplot(
            bench.runtime,
            bench.baseline_label,
            f"{bench.metric_label}: median runtime, pyvisim vs. {bench.baseline_label}",
            output_dir / f"{bench.slug}_runtime_barplot.png",
        )
    _inject_section(readme_path, _format_section(benchmarks, seed, repeats))
    print(f"Wrote {json_path}")
    print(f"Wrote {output_dir}/{{ssim,ms_ssim}}_runtime_barplot.png")
    print(f"Refreshed the Benchmark section of {readme_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    :param argv: Raw arguments, or ``None`` for ``sys.argv``.
    :return: The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "structural" / "benchmark",
        help="Directory of the JSON and PNG outputs.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=_REPO_ROOT / "docs" / "structural" / "README.md",
        help="Markdown file whose Benchmark section is refreshed.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the image sampling."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of timed calls per runtime workload.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run every benchmark and write the outputs.

    :param argv: Raw command-line arguments, or ``None`` for ``sys.argv``.
    """
    args = _parse_args(argv)
    os.environ["PYVISIM_NUM_THREADS"] = str(_NUM_THREADS)
    torch.set_num_threads(_NUM_THREADS)
    rng = np.random.default_rng(args.seed)
    dataset = OxfordFlowerDataset(purpose=_DATASET_SPLIT)
    # One accuracy group and four runtime groups per metric, all disjoint.
    counts_per_metric = [_N_ACCURACY_IMAGES, 2, 1, _SMALL_BATCH, _LARGE_BATCH]
    groups = _sample_disjoint_images(dataset, rng, counts_per_metric * 2)
    benchmarks = (
        _benchmark_ssim(groups[0], groups[1:5], rng, args.repeats),
        _benchmark_msssim(groups[5], groups[6:10], rng, args.repeats),
    )
    _write_outputs(benchmarks, args.output_dir, args.readme, args.seed, args.repeats)


if __name__ == "__main__":
    main()
