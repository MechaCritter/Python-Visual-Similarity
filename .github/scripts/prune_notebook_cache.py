"""
Remove cached notebook executions that no longer match a tutorial notebook.

The documentation build executes the tutorial notebooks through jupyter-cache,
which keys every execution on the code cells of its notebook. A changed notebook
adds a new execution and leaves the previous one behind, so a cache that CI
carries from run to run would grow with every change. This script keeps only the
executions that still match a notebook of the checkout::

    uv run python .github/scripts/prune_notebook_cache.py \\
        docs/_build/.jupyter_cache docs/tutorials/notebooks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_cache import get_cache
from jupyter_cache.base import JupyterCacheAbstract


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def matching_record_ids(cache: JupyterCacheAbstract, notebooks: Path) -> set[int]:
    """
    Collect the cache records that match a notebook of a folder.

    :param cache: The notebook execution cache.
    :param notebooks: Folder holding the ``.ipynb`` notebooks.
    :return: Primary keys of the records that match one of the notebooks.
    """
    record_ids = set()
    for path in sorted(notebooks.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        try:
            record_ids.add(cache.match_cache_notebook(notebook).pk)
        except KeyError:
            continue
    return record_ids


def remove_other_records(cache: JupyterCacheAbstract, keep: set[int]) -> int:
    """
    Remove every cache record whose primary key is not in ``keep``.

    :param cache: The notebook execution cache.
    :param keep: Primary keys of the records to keep.
    :return: The number of removed records.
    """
    stale = [
        record.pk for record in cache.list_cache_records() if record.pk not in keep
    ]
    for pk in stale:
        cache.remove_cache(pk)
    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune the notebook execution cache.")
    parser.add_argument("cache", type=Path, help="the jupyter-cache folder")
    parser.add_argument("notebooks", type=Path, help="the folder holding the notebooks")
    args = parser.parse_args()

    if not args.cache.is_dir():
        report(f"No notebook cache at {args.cache}, nothing to prune.")
        return

    cache = get_cache(args.cache)
    removed = remove_other_records(cache, matching_record_ids(cache, args.notebooks))
    report(f"Removed {removed} stale notebook execution(s) from {args.cache}.")


if __name__ == "__main__":
    main()
