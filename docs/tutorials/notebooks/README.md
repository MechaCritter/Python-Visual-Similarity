# Tutorial notebooks

Rules for working with the notebooks in this folder:

- Each notebook is rendered as a page of the documentation. Give it exactly one
  top-level heading (`# Title`) and nest every other heading below it, since
  every top-level heading becomes an item of its own in the sidebar.
- Commit notebooks without outputs. Run `make strip-notebooks` before you
  commit, and `make check-notebooks` to verify what the CI verifies.
- Keep the workload small enough to finish on a CPU in under 5 minutes. Otherwise, 
  adjust parameters such as dataset size, number of epochs,...
- Put shared helper functions in `tutorial_utils.py` instead of copying them
  between notebooks.
- Register a new notebook in the table and the `toctree` of its chapter page
  under `docs/tutorials/` (`1_introduction.rst`, `2_classical_methods.rst`, ...).
  The sidebar numbers it as `1.1 <heading>` from the notebook's title.
- Files a notebook writes next to itself (stores, checkpoints, `runs/`) are
  ignored by git and must stay out of the history.
- Install what the notebooks import with `uv sync --group tutorials --extra nn`,
  and run them all with `make test-notebooks`.
- `make docs` renders the notebooks as they are on disk and does not execute
  them. To see their outputs, run `make test-notebooks` first, which writes the
  executed notebooks back in place.
