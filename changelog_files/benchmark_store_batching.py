"""
Measure what the batched, prefetching gallery build of the image store buys.

The same gallery is built twice per round: once against the working tree, which
hands the embedder a whole batch at a time and reads the files on worker
threads, and once against a baseline revision, which reads and embeds one image
at a time.

Every stage runs in its own child process. The two versions of the library
therefore never share one interpreter, and the process orchestrating them holds
nothing but the standard library while a build runs for the better part of an
hour, which is why the library is imported inside the child stages rather than
at module level.

Only the store constructor is timed. The embedder is built, and the Fisher
Vector vocabulary is fitted, before the clock starts. Every finished
measurement is appended to a results file, so an interrupted run can be picked
up where it stopped.

Run it with::

    uv run --extra nn --group bench python changelog_files/benchmark_store_batching.py
"""

import argparse
import contextlib
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

_SCRIPT_REF = "changelog_files/benchmark_store_batching.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Split the gallery is drawn from. Its every image is used.
_DATASET_SPLIT = "train"
#: Index the gallery is built into, and the parameters it is built with.
_SEARCH_INDEX = "hnsw"
_INDEX_PARAMS = {"m": 16, "ef_construction": 200}

#: CLIP variant and weights the gallery is embedded with.
_CLIP_VARIANT = "ViT-B-32"
_CLIP_PRETRAINED = "openai"

#: Mixture components of the Fisher Vector vocabulary.
_N_COMPONENTS = 32
#: Factor the local descriptors are reduced by before the vocabulary is fitted.
_DIM_REDUCTION_FACTOR = 2
#: Images the vocabulary is fitted on, outside of every measurement.
_VOCABULARY_IMAGES = 100
#: Seed of the vocabulary image sampling and of the GMM fit.
_SEED = 0

#: Times every case is measured per version.
_ROUNDS = 1
#: Fixed thread budget of every run.
_NUM_THREADS = 4
#: Batch size for both embedders
_BATCH_SIZE = 32

#: Revision the working tree is compared against: the commit right before the
#: batched gallery build landed.
_BASELINE_REVISION = "HEAD~1"
#: Name of the version each variant reports under.
_BEFORE = "before"
_AFTER = "after"

#: Name of the fitted vocabulary file inside the workspace.
_EMBEDDER_FILE = "vocabulary.embedder"
#: Name of the gallery path file inside the workspace.
_PATHS_FILE = "paths.json"


def _log(message: str) -> None:
    """Print a progress line right away, so a long run can be followed."""
    print(message, flush=True)


@dataclass(frozen=True)
class Case:
    """
    One gallery build the benchmark times.

    :param name: Command-line name of the case.
    :param title: Name the case is reported under.
    :param embedder: Description of the embedder the gallery is built with.
    """

    name: str
    title: str
    embedder: str


#: The gallery builds that are measured, in the order they are run.
_CASES = (
    Case(
        name="clip",
        title="CLIP",
        embedder=f'ClipEmbedder("{_CLIP_VARIANT}", "{_CLIP_PRETRAINED}", batch_size={_BATCH_SIZE})',
    ),
    Case(
        name="fisher",
        title="Fisher Vector",
        embedder=(
            f"FisherVectorEmbedder(n_components={_N_COMPONENTS}, batch_size={_BATCH_SIZE})`, fitted with "
            f"`learn(images, dim_reduction_factor={_DIM_REDUCTION_FACTOR})"
        ),
    ),
)


@dataclass
class Measurement:
    """
    What one case cost on one version of the library.

    :param case: Name of the case that was measured.
    :param variant: Version the case ran against, ``"before"`` or ``"after"``.
    :param rounds: Wall-clock duration of every round, in seconds.
    """

    case: str
    variant: str
    rounds: list[float] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        """Median duration of the rounds, in seconds."""
        return statistics.median(self.rounds)


def _prepare(args: argparse.Namespace) -> None:
    """
    Write the gallery paths out and fit the Fisher Vector vocabulary.

    This is the first stage the orchestrator spawns. It runs against the
    working tree and leaves both versions the exact same inputs to work from.

    :param args: The parsed command-line arguments.
    """
    from pyvisim.datasets import OxfordFlowerDataset

    paths = OxfordFlowerDataset(purpose=_DATASET_SPLIT).image_paths
    if args.num_images is not None:
        paths = paths[: args.num_images]
    args.paths_file.write_text(json.dumps(paths), encoding="utf-8")

    _log(f"Warming the page cache with {len(paths)} images.")
    _warm_page_cache(paths)

    if args.embedder_file is None:
        return
    images = _vocabulary_images(paths, _VOCABULARY_IMAGES, _SEED)
    _log(f"Fitting the vocabulary on {len(images)} images.")
    _fit_vocabulary(images, args.embedder_file)


def _warm_page_cache(paths: Sequence[str]) -> None:
    """
    Read every gallery file once, before any measurement.

    A first read pulls the file off the disk, every later one finds it in the
    page cache of the operating system. Warming them all up front keeps that
    one-off cost out of the measurements, which compare how the two versions
    read and embed the gallery rather than how fast the disk is.
    """
    for path in paths:
        with open(path, "rb") as handle:
            handle.read()


def _vocabulary_images(paths: Sequence[str], count: int, seed: int) -> list[Any]:
    """
    Load the images the Fisher Vector vocabulary is fitted on.

    :param paths: Gallery image paths to draw from.
    :param count: Number of images to load.
    :param seed: Seed of the image sampling.
    :return: The sampled images, as canonical RGB arrays.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=min(count, len(paths)), replace=False)
    images = []
    for index in indices:
        with Image.open(paths[int(index)]) as image:
            images.append(np.asarray(image.convert("RGB")))
    return images


def _fit_vocabulary(images: list[Any], destination: Path) -> None:
    """
    Fit the Fisher Vector vocabulary once and write the embedder to disk.

    Both versions load the same file, so they embed the gallery through the
    exact same vocabulary and the training stays out of the measurements.

    :param images: Images the vocabulary is fitted on.
    :param destination: File the fitted embedder is written to.
    """
    from pyvisim.classic import FisherVectorEmbedder

    embedder = FisherVectorEmbedder(
        n_components=_N_COMPONENTS, gmm_params={"rng": _SEED}, batch_size=64
    )
    embedder.learn(images, dim_reduction_factor=_DIM_REDUCTION_FACTOR)
    embedder.save_to_disk(destination)


def _build_embedder(case: str, embedder_file: Path | None) -> Any:
    """
    Build the embedder of one case, outside of the timed section.

    :param case: Name of the case to build the embedder of.
    :param embedder_file: File holding the fitted Fisher Vector embedder.
    :return: The ready-to-use embedder.
    :raises ValueError: If the case is unknown, or the Fisher Vector case is
        asked for without a fitted embedder.
    """
    if case == "clip":
        from pyvisim.neural_networks import ClipEmbedder

        return ClipEmbedder(_CLIP_VARIANT, pretrained=_CLIP_PRETRAINED, batch_size=64)
    if case == "fisher":
        from pyvisim.classic import FisherVectorEmbedder

        if embedder_file is None:
            raise ValueError("The Fisher Vector case needs a fitted embedder file.")
        return FisherVectorEmbedder.load_from_disk(str(embedder_file))
    raise ValueError(f"Unknown case {case!r}.")


def _build_store(paths: list[str], embedder: Any) -> float:
    """
    Build the gallery store once, and return how long it took.

    :param paths: Gallery image paths the store is built over.
    :param embedder: Embedder the gallery is embedded with.
    :return: The wall-clock duration of the build, in seconds.
    """
    from pyvisim.image_store import InMemoryImageEmbeddingStore

    gc.collect()
    start = time.perf_counter()
    InMemoryImageEmbeddingStore(
        image_paths=paths,
        embedder=embedder,
        search_index=_SEARCH_INDEX,
        index_params=dict(_INDEX_PARAMS),
    )
    return time.perf_counter() - start


def _measure(args: argparse.Namespace) -> None:
    """
    Run a single timed build and print its result as one JSON line.

    This is the stage the orchestrator spawns per measurement, against one
    version of the library at a time.

    :param args: The parsed command-line arguments.
    :raises RuntimeError: If the imported library is not the expected one, or a
        GPU is visible, which would time something else than the CPU build.
    :raises ValueError: If the internal arguments of the stage are missing.
    """
    import torch

    import pyvisim

    if args.paths_file is None or args.expect_tree is None:
        raise ValueError("'--measure' needs '--paths-file' and '--expect-tree'.")
    tree = Path(pyvisim.__file__).resolve().parents[1]
    if tree != args.expect_tree.resolve():
        raise RuntimeError(
            f"Imported pyvisim from {str(tree)!r}, expected "
            f"{str(args.expect_tree.resolve())!r}."
        )
    if torch.cuda.is_available():
        raise RuntimeError("The GPU is still visible; both versions run on the CPU.")
    torch.set_num_threads(_NUM_THREADS)

    paths = json.loads(args.paths_file.read_text(encoding="utf-8"))
    embedder = _build_embedder(args.measure, args.embedder_file)
    seconds = _build_store(paths, embedder)
    print(json.dumps({"case": args.measure, "seconds": seconds}), flush=True)


def _extension_sources(tree: Path) -> dict[str, bytes]:
    """Read the Cython sources of a tree, keyed by their relative path."""
    package = tree / "pyvisim"
    return {
        str(source.relative_to(tree)): source.read_bytes()
        for pattern in ("*.pyx", "*.pxd")
        for source in package.rglob(pattern)
    }


def _copy_built_extensions(tree: Path) -> None:
    """
    Copy the compiled extension modules of the working tree into a worktree.

    A fresh worktree holds no build artifacts, and building them again per
    revision would compare two differently built libraries. The copies are only
    valid while both trees carry the same Cython sources, which is checked
    first.

    :param tree: Worktree the compiled modules are copied into.
    :raises RuntimeError: If the Cython sources of the two trees differ.
    """
    if _extension_sources(tree) != _extension_sources(_REPO_ROOT):
        raise RuntimeError(
            "The Cython sources of the baseline revision differ from the ones "
            "of the working tree, so its compiled modules cannot be reused. "
            "Build the extensions of the baseline worktree instead."
        )
    for built in sorted((_REPO_ROOT / "pyvisim").rglob("*.so")):
        destination = tree / built.relative_to(_REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, destination)


@contextlib.contextmanager
def _baseline_tree(revision: str) -> Iterator[Path]:
    """
    Check the baseline revision out into a throw-away git worktree.

    :param revision: Revision the working tree is compared against.
    :return: A context manager yielding the path of the worktree.
    """
    with tempfile.TemporaryDirectory(prefix="pyvisim-baseline-") as directory:
        tree = Path(directory) / "tree"
        _run_git("worktree", "add", "--detach", str(tree), revision)
        try:
            _copy_built_extensions(tree)
            yield tree
        finally:
            _run_git("worktree", "remove", "--force", str(tree))


def _run_git(*arguments: str) -> str:
    """Run a git command in the repository and return its output."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_stage(tree: Path, arguments: Sequence[str]) -> str:
    """
    Spawn one stage of the benchmark against one version of the library.

    ``PYTHONPATH`` puts the tree under measurement ahead of the installed
    package, so the child imports the version it is meant to run.

    :param tree: Tree holding the version of the library to run against.
    :param arguments: Command-line arguments of the stage.
    :return: The standard output of the child process.
    """
    environment = dict(
        os.environ,
        PYTHONPATH=str(tree),
        PYVISIM_NUM_THREADS=str(_NUM_THREADS),
        CUDA_VISIBLE_DEVICES="",
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=tree,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _run_measurement(
    tree: Path,
    case: str,
    paths_file: Path,
    embedder_file: Path | None,
) -> float:
    """
    Spawn one timed build and read its duration back.

    :param tree: Tree holding the version of the library to measure.
    :param case: Name of the case to measure.
    :param paths_file: File holding the gallery image paths as JSON.
    :param embedder_file: File holding the fitted Fisher Vector embedder.
    :return: The wall-clock duration of the build, in seconds.
    :raises RuntimeError: If the child process reports no measurement.
    """
    arguments = [
        "--measure",
        case,
        "--paths-file",
        str(paths_file),
        "--expect-tree",
        str(tree),
    ]
    if embedder_file is not None:
        arguments += ["--embedder-file", str(embedder_file)]
    for line in reversed(_run_stage(tree, arguments).splitlines()):
        if line.startswith("{"):
            return float(json.loads(line)["seconds"])
    raise RuntimeError(f"The measurement of {case!r} printed no result.")


def _load_measurements(
    results_file: Path,
    cases: Sequence[Case],
    variants: Sequence[str],
) -> dict[tuple[str, str], Measurement]:
    """
    Start from an empty set of measurements, or from the recorded ones.

    :param results_file: File the finished rounds are recorded in.
    :param cases: The gallery builds that are measured.
    :param variants: The versions every case is measured against.
    :return: One measurement per case and version, holding the recorded rounds.
    """
    measurements = {
        (case.name, variant): Measurement(case.name, variant)
        for case in cases
        for variant in variants
    }
    if not results_file.exists():
        return measurements
    for line in results_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["case"], record["variant"])
        if key in measurements:
            measurements[key].rounds.append(float(record["seconds"]))
    return measurements


def _record(results_file: Path, case: str, variant: str, seconds: float) -> None:
    """Append one finished round to the results file."""
    record = json.dumps({"case": case, "variant": variant, "seconds": seconds})
    with open(results_file, "a", encoding="utf-8") as handle:
        handle.write(f"{record}\n")


def _measure_all(
    cases: Sequence[Case],
    trees: dict[str, Path],
    rounds: int,
    files: tuple[Path, Path | None, Path],
) -> dict[tuple[str, str], Measurement]:
    """
    Measure every case against both versions, once per round.

    The versions are interleaved across the rounds, so that clock drift and
    thermal throttling reach both of them alike instead of biasing whichever
    one would have run last. Rounds already recorded in the results file are
    not measured again.

    :param cases: The gallery builds to measure.
    :param trees: The tree of each version, keyed by its name.
    :param rounds: Times every case is measured per version.
    :param files: The gallery path file, the fitted embedder file and the
        results file, in that order.
    :return: One measurement per case and version.
    """
    paths_file, embedder_file, results_file = files
    measurements = _load_measurements(results_file, cases, list(trees))
    for round_index in range(rounds):
        for case in cases:
            for variant, tree in trees.items():
                measurement = measurements[case.name, variant]
                if len(measurement.rounds) > round_index:
                    _log(
                        f"round {round_index + 1}/{rounds}, {case.name} "
                        f"{variant}: recorded, skipping"
                    )
                    continue
                seconds = _run_measurement(
                    tree,
                    case.name,
                    paths_file,
                    embedder_file if case.name == "fisher" else None,
                )
                measurement.rounds.append(seconds)
                _record(results_file, case.name, variant, seconds)
                _log(
                    f"round {round_index + 1}/{rounds}, {case.name} "
                    f"{variant}: {seconds:.1f}s"
                )
    return measurements


def _format_seconds(seconds: float) -> str:
    """Format a duration the way the report reads it."""
    return f"{seconds:.0f} s"


def _format_table(
    cases: Sequence[Case],
    measurements: dict[tuple[str, str], Measurement],
) -> str:
    """Render the runtimes of every case as one markdown table per case."""
    blocks = []
    for case in cases:
        before = measurements[case.name, _BEFORE].seconds
        after = measurements[case.name, _AFTER].seconds
        blocks.append(
            f"### {case.title}\n\n"
            "| Before | After |\n"
            "|---|---|\n"
            f"| {_format_seconds(before)} | {_format_seconds(after)} |"
        )
    return "\n\n".join(blocks)


def _package_version(name: str) -> str:
    """Resolve an installed package version, or ``"unknown"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_report(
    cases: Sequence[Case],
    measurements: dict[tuple[str, str], Measurement],
    count: int,
    rounds: int,
    baseline: str,
) -> str:
    """Render the complete markdown report."""
    embedders = "\n".join(f"- {case.title}: `{case.embedder}`" for case in cases)
    return f"""# Store build benchmark, before and after the batched gallery build

> [!IMPORTANT]
> This file was generated by the script [`{_SCRIPT_REF}`](../../{_SCRIPT_REF}).
> **Do not edit manually!**

All {count} images of the Oxford Flower dataset (`{_DATASET_SPLIT}` split) are
built into an `InMemoryImageEmbeddingStore` on the `{_SEARCH_INDEX}` index with
`{_INDEX_PARAMS}`, once per version of the library. **Before** is the revision
`{baseline}`, which reads and embeds one image at a time; **after** is the
working tree, which reads the files on worker threads and hands the embedder a
whole batch at a time.

The embedders are:

{embedders}

Only the store constructor is timed: the CLIP weights are loaded and the Fisher
Vector vocabulary is fitted before the clock starts, and both versions load the
same fitted embedder from disk. Every case is measured {rounds} time(s) per
version, with the versions interleaved across the rounds so that clock drift
reaches both of them alike. Every file is read once before the first
measurement, which leaves the images in the page cache of the operating system.
Both versions run on the CPU, on the same thread budget.

## Measurements

{_format_table(cases, measurements)}

## Environment

| | |
|---|---|
| pyvisim | {_package_version("pyvisim")} |
| Python | {platform.python_version()} |
| NumPy | {_package_version("numpy")} |
| Pillow | {_package_version("pillow")} |
| PyTorch | {_package_version("torch")} |
| Platform | {platform.platform()} |
| CPUs | {os.cpu_count()} |
| Threads | `PYVISIM_NUM_THREADS={_NUM_THREADS}` |
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure how long InMemoryImageEmbeddingStore takes to "
        "build a gallery, before and after its batched and prefetching build."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "benchmarks" / "store_batching.md",
        help="Markdown file the report is written to.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="File the finished rounds are recorded in, so that an interrupted "
        "run can be resumed. Defaults to the output path with a '.jsonl' suffix.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Measure every round again instead of resuming the recorded ones.",
    )
    parser.add_argument(
        "--baseline-rev",
        default=_BASELINE_REVISION,
        help="Revision the working tree is compared against.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in _CASES],
        default=[case.name for case in _CASES],
        help="Gallery builds to measure.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=_ROUNDS,
        help="Times every case is measured per version.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Gallery images to build the store over. Defaults to the whole split.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Internal: collect the gallery paths and fit the vocabulary.",
    )
    parser.add_argument(
        "--measure",
        choices=[case.name for case in _CASES],
        help="Internal: run a single timed build and print it as JSON.",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        help="Internal: file holding the gallery image paths as JSON.",
    )
    parser.add_argument(
        "--embedder-file",
        type=Path,
        help="Internal: file holding the fitted Fisher Vector embedder.",
    )
    parser.add_argument(
        "--expect-tree",
        type=Path,
        help="Internal: tree the measured library has to be imported from.",
    )
    return parser.parse_args(argv)


def _results_file(args: argparse.Namespace) -> Path:
    """Resolve the results file, emptied when the run is asked to be fresh."""
    path = args.results_file or args.output.with_suffix(".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and path.exists():
        path.unlink()
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Measure both versions of the gallery build and write the report."""
    args = _parse_args(argv)
    if args.prepare:
        _prepare(args)
        return
    if args.measure is not None:
        _measure(args)
        return

    cases = [case for case in _CASES if case.name in args.cases]
    results_file = _results_file(args)
    with tempfile.TemporaryDirectory(prefix="pyvisim-benchmark-") as directory:
        workspace = Path(directory)
        paths_file = workspace / _PATHS_FILE
        embedder_file = None
        if any(case.name == "fisher" for case in cases):
            embedder_file = workspace / _EMBEDDER_FILE

        arguments = ["--prepare", "--paths-file", str(paths_file)]
        if embedder_file is not None:
            arguments += ["--embedder-file", str(embedder_file)]
        if args.num_images is not None:
            arguments += ["--num-images", str(args.num_images)]
        print(_run_stage(_REPO_ROOT, arguments), end="", flush=True)
        count = len(json.loads(paths_file.read_text(encoding="utf-8")))

        with _baseline_tree(args.baseline_rev) as baseline:
            measurements = _measure_all(
                cases,
                {_BEFORE: baseline, _AFTER: _REPO_ROOT},
                args.rounds,
                (paths_file, embedder_file, results_file),
            )

    report = _format_report(
        cases,
        measurements,
        count,
        args.rounds,
        _run_git("rev-parse", "--short", args.baseline_rev),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    _log(f"Wrote {args.output}")
    print(_format_table(cases, measurements))


if __name__ == "__main__":
    main()
