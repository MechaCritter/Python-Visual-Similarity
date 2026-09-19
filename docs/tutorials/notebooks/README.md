# Tutorial notebooks

The tutorials are Jupyter notebooks in this folder. Sphinx renders every
notebook as a page of the documentation, grouped into numbered chapters.

## Setup

1. Install the dependencies with `uv sync --group tutorials --extra nn`.
2. Put (and reuse any) shared helper functions in `tutorial_utils.py`, when applicable.

## Writing a notebook

- Give each notebook exactly one top-level heading (`# Title`) and nest every
  other heading below it, since every top-level heading becomes an item of its
  own in the sidebar.
- Register a new notebook in the table and the `toctree` of its chapter page
  under `docs/tutorials/`.
- Keep the workload small enough to finish on a CPU in under 5 minutes, since
  GitHub's runners have no GPU. Otherwise, adjust parameters such as the dataset
  size or the number of epochs.
- Files a notebook writes next to itself (stores, checkpoints, `runs/`) are
  ignored by git and must stay out of the history.


## Running and rendering

- Use `make test-notebooks` to check if all the tutorials pass with your changes. This runs all notebooks in parallel from scratch. Notebooks that share one
  GPU can run out of memory, so limit the parallel runs with `make test-notebooks NOTEBOOK_WORKERS=2`.
- You can run the notebooks yourself and then `make docs` to see how the final
  documentation would look like before commmiting. Note that the outputs need to be stripped before committing, as the notebooks will be rerun in the CI and the outputs will be regenerated.
- To emulate the behavior of the CI, use `make docs NB_EXECUTION_MODE=cache` which executes all notebooks, caches their outputs in `docs/_build/.jupyter_cache`, and then generate the docs. If you re-execute this command, the notebooks will be executed again only if at least a code cell has changed since the last execution. If you made changes that are not reflected in the notebooks, purge the cache by deleting the `docs/_build/` folder and re-run `make docs NB_EXECUTION_MODE=cache`.  

## CI

> [!IMPORTANT]
>
> Run the command `make strip-notebooks` before you commit the notebooks to 
strip all the outputs and kernel metadata to prevent notebooks from bloating the git history. You can download the rendered HTML to review the created docs. Follow these steps:
> - As the workflow `Docs / Build documentation` finishes, on the GitHub page of your PR, click on it, then go to `Summary` at the top left. 
> - Under `Artifacts`, click on `docs-html` to download the generated final documentation. 
> - Unzip it and open `index.html` to review the documentation.

In order that the CI does not run too slow for every single PR, notebooks are
cached on CI and only re-run if a notebook has changed since the last run. This
of course has the weakness that a notebook can break if the library changes
underneath it. Hence, the `Tutorials` workflow runs every notebook from
scratch every three days, since a notebook can also break when the library
changes underneath it.

